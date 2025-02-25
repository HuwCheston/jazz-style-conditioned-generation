#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Parse potential matches from blindfold test corpus using fuzzy string matching and structured LLM outputs"""

import json
import os
from itertools import product
from typing import Optional

import pandas as pd
from dotenv import find_dotenv, load_dotenv
from joblib import Parallel, delayed
from loguru import logger
from openai import OpenAI
from pydantic import BaseModel, Field
from thefuzz import fuzz
from tqdm import tqdm

from jazz_style_conditioned_generation import utils

WINDOW_SIZE = 300
WINDOW_STRIDE = WINDOW_SIZE // 2  # 50% overlap between windows

PERFORMER_MATCH_THRESH = 90  # We enforce a strong match for performers
TRACK_ALBUM_MATCH_THRESH = 70  # We require a less strong match for tracks or albums

CSV_PATH = os.path.join(utils.get_project_root(), "outputs/blindfold_tests/potential_matches.csv")
FEWSHOT_FP = os.path.join(utils.get_project_root(), "outputs/blindfold_tests/fewshot_prompts.json")
FEWSHOT_EXAMPLES = utils.read_json_cached(FEWSHOT_FP)

MIN_CHARS_FOR_MATCH = 7
MATCH_FUNC = fuzz.partial_ratio

OAI_MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = """You are an expert at structured data extraction. You will be given unstructured text from an 
interview with a famous musician and should convert it into the given structure. The text you will read comes from a 
'Blindfold Test' interview printed in the 'Downbeat' jazz magazine. The 'Blindfold Test' is a listening test that 
challenges the featured artist to discuss and identify the music and musicians who performed on selected recordings. 
The artist is then asked to rate each tune using a 5-star system. No information is given to the artist prior to the 
test."""


class InterviewPerformance(BaseModel):
    """Information about a reference to a single performance in the interview"""
    performance_in_text: bool = Field(
        ...,
        description="Whether or not the interview references the given performance"
    )
    text_requires_cleaning: Optional[bool] = Field(
        None,
        description="Whether or not the text requires spelling and typographical cleaning"
    )
    section_of_text: Optional[str] = Field(
        None,
        description="The section of the interview that references the given performance, with spelling and "
                    "typographical errors corrected but with the content unchanged.",
    )


class Interview(BaseModel):
    """Information about a complete interview"""
    performances: list[InterviewPerformance] = Field(
        ...,
        description="A single item for every given performance",
        min_length=1
    )


def get_oai_client() -> OpenAI:
    """Returns an OpenAI client for interview descriptions using environment variables"""
    load_dotenv(find_dotenv())  # sorry, I'm not leaking my API key ;)
    return OpenAI(
        organization=os.environ["OPENAI_ORG"],
        project=os.environ["OPENAI_PROJECT"],
        api_key=os.environ["OPENAI_API_KEY"],
    )


def clean_text(text: str) -> str:
    """Cleans test by replacing whitespace and newline characters"""
    return text.strip().replace('\n', ' ').replace('  ', ' ')


def read_text(text_fpath: str) -> dict:
    """Reads a text file with a cache to prevent redundant operations"""
    with open(text_fpath, 'r') as lo:
        loaded = clean_text(lo.read())
    return loaded


def check_publication_date_against_recording_date(
        downbeat_filepath: str,
        recording_year: str | int | None
) -> bool:
    """Returns True if the recording was made before the downbeat magazine was published, False if not"""
    # For some tracks we don't have the recording year, so we should process these for safety
    if recording_year is None:
        return True
    # If we have a recording year for this track (this is the case for all but ~200/4000 tracks)
    else:
        if isinstance(recording_year, str):
            recording_year = eval(recording_year)
        # Parse the year the magazine was published from the filepath
        downbeat_year = eval(os.path.basename(downbeat_filepath).split('-')[1])
        # If the recording was made after the magazine was published, it cannot be a match, so just return
        if recording_year > downbeat_year:
            return False
        else:
            return True


def matcher(text: str, str_to_match: str) -> float:
    """Applies MATCH_FUNC between input text and desired string if desired string exceeds a minimum length"""
    if len(str_to_match) >= MIN_CHARS_FOR_MATCH:
        return MATCH_FUNC(text, str_to_match)
    else:
        return 0.


def process_parallel(
        blindfold_fp: str,
        track_name: str,
        album_name: str,
        performer_name: str,
        recording_year: str | int | None,
        filepath: str
) -> list[dict]:
    """Parallel processing through windowed sections of blindfold tests"""
    # If the recording was made AFTER the downbeat magazine was published, it cannot be a match
    if check_publication_date_against_recording_date(blindfold_fp, recording_year) is False:
        # So, we can just return an empty list straight away: this will be filtered out
        return []

    # Lead the interview text from the filepath
    loaded = read_text(blindfold_fp)
    res = []
    # Iterate over windowed sections of the text
    for window_start in range(0, len(loaded) - WINDOW_SIZE, WINDOW_STRIDE):
        # Subset the text for the current window
        window_end = window_start + WINDOW_SIZE
        window = loaded[window_start:window_end]
        # Break out once we don't have enough characters
        if len(window) < WINDOW_SIZE:
            break
        # Compute matches with all metadata
        track_match = matcher(window, track_name)
        album_match = matcher(window, album_name)
        performer_match = matcher(window, performer_name)
        # If we meet all criteria
        if performer_match >= PERFORMER_MATCH_THRESH:
            if album_match >= TRACK_ALBUM_MATCH_THRESH or track_match >= TRACK_ALBUM_MATCH_THRESH:
                # Then we can append the results to our list
                res.append(dict(
                    blindfold_fp=blindfold_fp,
                    text=window,
                    track_name=track_name,
                    album_name=album_name,
                    performer_name=performer_name,
                    track_match=track_match,
                    album_match=album_match,
                    performer_match=performer_match,
                    total_match=track_match + album_match + performer_match,
                    track_fp=filepath,
                ))
    return res


def get_preliminary_matches(metadata: list[dict]) -> list[dict]:
    """Gets preliminary matches between downbeat interviews and database tracks using fuzzy string matching"""
    try:
        # If we've already processed this information, we can just grab the CSV file directly
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        # Get the metadata fields we need from each track
        names = [
            (m["fname"], m["track_name"], m["album_name"], m["pianist"], m["recording_year"])
            for m in metadata
        ]
        # Get the filepaths for all blindfold test interviews
        bts_fp = utils.get_data_files_with_ext(dir_from_root="data/blindfold_tests", ext="**/*.txt")
        # Get combinations between dataset tracks + interviews: len = N(database_tracks) * N(interviews)
        combs = list(product(bts_fp, names))
        # Process each interview and track combination in parallel
        with Parallel(n_jobs=-1, verbose=0) as par:
            proc = par(
                delayed(process_parallel)(b, t, a, p, r, f) for b, (f, t, a, p, r) in tqdm(combs, desc="Processing...")
            )
        # Remove empty results and convert to a flat list
        sub = [x for xs in proc for x in xs if len(x) > 0]
        # Convert to a dataframe and save as a CSV so we don't have to repeat this processing
        df = pd.DataFrame(sub).sort_values(by="total_match", ascending=False)
        df.to_csv(CSV_PATH)
    return (
        # If the same track matches to the same interview multiple times, we don't want to make any redundant calls
        df.drop_duplicates(subset=["blindfold_fp", "track_fp"])
        .sort_values(by="total_match", ascending=False)
        .reset_index(drop=True)
        .to_dict(orient="records")
    )


def create_prompt(match_dict: dict) -> list[dict]:
    """For a potential match between a database track and interview, create the prompt we'll pass to the LLM"""
    interview_text = read_text(match_dict["blindfold_fp"])
    performer, track, album = match_dict["performer_name"], match_dict["track_name"], match_dict["album_name"]
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        *FEWSHOT_EXAMPLES,
        {
            "role": "user",
            "content": f"Extract the section of text in this interview that refers to the performance by '{performer}' "
                       f"with title '{track}' from album: '{album}'."
        },
        {
            "role": "user",
            "content": interview_text
        }
    ]


def prompt_llm(client: OpenAI, prompt: list[dict]) -> dict:
    """Prompt the LLM with a structured output format and parse as a dictionary"""
    completion = client.beta.chat.completions.parse(
        model=OAI_MODEL,
        messages=prompt,
        response_format=InterviewPerformance
    )
    return completion.choices[0].message.parsed.model_dump(mode="json")


def main():
    metadata_fps = utils.get_data_files_with_ext("data/raw", "**/*_tivo.json")
    metadata = [utils.read_json_cached(js) for js in metadata_fps]
    preliminary_matches = get_preliminary_matches(metadata)
    logger.info(f'Got {len(preliminary_matches)} preliminary matches!')
    client = get_oai_client()
    all_responses = []
    for match in tqdm(preliminary_matches, desc="Prompting LLM with preliminary matches..."):
        outpath = f"{match['blindfold_fp'].split('.')[0].replace('/data/blindfold_tests', '/outputs/blindfold_tests/llm_parsing')}_{match['track_fp']}.json"
        if os.path.exists(outpath):
            response = utils.read_json_cached(outpath)
        else:
            prompt = create_prompt(match)
            response = prompt_llm(client, prompt)
            # Finalise response by adding missing key-value pairs in
            for key, value in match.items():
                if key not in response.keys():
                    response[key] = value
            with open(outpath, "w") as op:
                json.dump(response, op, indent=4, ensure_ascii=False)
        all_responses.append(response)
    logger.info(f'Dumped {len(all_responses)} individual responses!')
    df = pd.DataFrame(all_responses)
    df.to_csv(os.path.join(utils.get_project_root(), "outputs/blindfold_tests/for_eval.csv"))
    logger.info('These responses should now be checked through manually to confirm whether they are correct.')


if __name__ == "__main__":
    main()
