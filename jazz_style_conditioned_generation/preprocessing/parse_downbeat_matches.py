#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Parse potential matches from downbeat corpus using a combination of string matching and structured LLM outputs"""

import json
import os
from itertools import product
from pathlib import Path
from typing import Union

import pandas as pd
from dotenv import find_dotenv, load_dotenv
from joblib import Parallel, delayed
from loguru import logger
from openai import OpenAI
from pydantic import BaseModel, Field
from thefuzz import fuzz
from tqdm import tqdm

from jazz_style_conditioned_generation import utils

WINDOW_SIZE = 200
WINDOW_STRIDE = 100
THRESH = 90

CSV_PATH = os.path.join(Path(os.path.abspath(os.curdir)).parent.parent, "outputs/blindfold_test_matches.csv")

MIN_CHARS_FOR_MATCH = 7
MATCH_FUNC = fuzz.partial_ratio

OAI_MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = """You are an expert at structured data extraction. You will be given unstructured text from an 
interview with a famous musician and should convert it into the given structure. The text you will read comes from a 
'Blindfold Test' interview printed in the 'Downbeat' jazz magazine. The 'Blindfold Test' is a listening test that 
challenges the featured artist to discuss and identify the music and musicians who performed on selected recordings. 
The artist is then asked to rate each tune using a 5-star system. No information is given to the artist prior to the 
test."""


class InterviewDescription(BaseModel):
    """Information about a reference to a single performance in the interview"""
    performance_in_text: bool = Field(
        ...,
        description="Whether or not the interview references the given performance"
    )
    text_requires_cleaning: Union[bool, None] = Field(
        None,
        description="Whether or not the text requires spelling and typographical cleaning"
    )
    section_of_text: Union[str, None] = Field(
        None,
        description="The section of the interview that references the given performance, with spelling and "
                    "typographical errors corrected but with the content unchanged.",
    )


def get_oai_client() -> OpenAI:
    load_dotenv(find_dotenv())
    return OpenAI(
        organization=os.environ["OPENAI_ORG"],
        project=os.environ["OPENAI_PROJECT"],
        api_key=os.environ["OPENAI_API_KEY"],
    )


def get_files(dir_from_root: str, ext: str) -> list[str]:
    root = Path(os.path.abspath(os.curdir)).parent.parent
    return [str(i) for i in Path(os.path.join(str(root), dir_from_root)).glob(ext)]


def clean_text(text: str):
    return text.strip().replace('\n', ' ').replace('  ', ' ')


def read_text(text_fpath: str) -> dict:
    """Reads a text file with a cache to prevent redundant operations"""
    with open(text_fpath, 'r') as lo:
        loaded = clean_text(lo.read())
    return loaded


def process_parallel(
        blindfold_fp: list[str],
        track_name: str,
        album_name: str,
        performer_name: str,
        filepath: str
) -> list[dict]:
    """Parallel processing through windowed sections of blindfold tests"""

    def matcher(win, to_match):
        if len(to_match) >= MIN_CHARS_FOR_MATCH:
            return MATCH_FUNC(win, to_match)
        else:
            return 0

    loaded = read_text(blindfold_fp)
    res = []

    for window_start in range(0, len(loaded) - WINDOW_SIZE, WINDOW_STRIDE):
        window_end = window_start + WINDOW_SIZE
        window = loaded[window_start:window_end]
        if len(window) < WINDOW_SIZE:
            break

        track_match = matcher(window, track_name)
        album_match = matcher(window, album_name)
        performer_match = matcher(window, performer_name)

        if any([pr > THRESH for pr in [track_match, album_match, performer_match]]):
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
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        names = [(m["fname"], m["track_name"], m["album_name"], m["pianist"]) for m in metadata]
        bts_fp = get_files(dir_from_root="data/blindfold_tests", ext="**/*.txt")
        with Parallel(n_jobs=-1, verbose=0) as par:
            combs = list(product(bts_fp, names))
            proc = par(
                delayed(process_parallel)(b, t, a, p, f) for b, (f, t, a, p) in tqdm(combs, desc="Processing..."))
        sub = [x for xs in proc for x in xs if len(x) > 0]
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
    interview_text = read_text(match_dict["blindfold_fp"])
    performer, track, album = match_dict["performer_name"], match_dict["track_name"], match_dict["album_name"]
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": f"Extract the section of text in this interview that refers to the performance by '{performer}',"
                       f" title: '{track}', album: '{album}'. Structure your response according "
                       "to the format of the provided JSON. If you find text corresponding to this performance, "
                       "set the `performance_in_text` boolean to `True` and clean any grammatical or typographical "
                       "errors in the text, but do not change the content. If you do not find text, "
                       "set the `performance_in_text` boolean to `False`."
        },
        {
            "role": "user",
            "content": interview_text
        }
    ]


def prompt_llm(client: OpenAI, prompt: list[dict]) -> dict:
    completion = client.beta.chat.completions.parse(
        model=OAI_MODEL,
        messages=prompt,
        response_format=InterviewDescription
    )
    return completion.choices[0].message.parsed.model_dump(mode="json")


def main():
    metadata_fps = get_files("data/raw", "**/*_tivo.json")
    metadata = [utils.read_json_cached(js) for js in metadata_fps]
    preliminary_matches = get_preliminary_matches(metadata)[:100]
    logger.info(f'Got {len(preliminary_matches)} preliminary matches!')
    client = get_oai_client()
    all_responses = []
    for match in tqdm(preliminary_matches, desc="Prompting LLM with preliminary matches..."):
        outpath = f"{match['blindfold_fp'].split('.')[0].replace('/data/', '/outputs/')}_{match['track_fp']}.json"
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
    logger.info(f'Dumped {len(all_responses)} responses to ./data/blindfold_tests!')


if __name__ == "__main__":
    main()
