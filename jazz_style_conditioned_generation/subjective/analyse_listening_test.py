#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Analyse results of the subjective listening test"""

# Results should be dumped from PsyNet/Dallinger as a JSON and saved as ./references/dallinger-export.json

import ast
import os
import random
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from jazz_style_conditioned_generation import utils, plotting

EXPORT_JSON = os.path.join(utils.get_project_root(), "references/dallinger-export.json")
FIGURES_DIR = os.path.join(utils.get_project_root(), "outputs/figures/listening_test")
if not os.path.isdir(FIGURES_DIR):
    os.makedirs(FIGURES_DIR)

# These are the different types of metadata/demographic questions we ask participants
# These are numeric types, so we should report average + SD
METADATA_NUMERIC_TYPES = ["age", "years_of_formal_training", "hours_of_daily_music_listening"]
# These are categorical types, so we should report counts + proportions
METADATA_CATEGORICAL_TYPES = ["gender", "country_of_birth", "country_of_residence", "money_from_playing_music",
                              "jazz_experience"]
# These are plain old feedback types, so we should just report individual strings
METADATA_FEEDBACK_TYPES = ["recognise_feedback", "similarity_feedback", "feedback"]


def coerce_type(dangerous: Any) -> Any:
    """Coerces an input to the required type: NoneType, Int, Float, or String returned"""
    # If should be None, return NoneType
    if dangerous in ["null", "None", None, "nan", "na"]:
        return None
    # If countable, return countable
    elif isinstance(dangerous, (int, float)):
        return dangerous
    # If string or something else, try and convert to int
    else:
        try:
            return float(dangerous)
        except (ValueError, TypeError):
            return dangerous


def format_answer(answer_dict: dict) -> dict:
    """Formats the nested JSON with key `answer` for `question`: `rating`"""
    answer_fmt = {}
    for k, v in answer_dict.items():
        # Unpack nested dictionary
        if isinstance(v, dict):
            v = v["null"]
        if isinstance(v, str):
            v = coerce_type(v)
        if k == "genre":
            k = "predicted_genre"
        answer_fmt[k] = v
    return answer_fmt


def format_metadata(metadata_dict: dict) -> dict:
    """Formats the nested JSON with key `metadata_` for `question`: `rating`"""
    # We only need the prompt dictionary, so grab that
    prompt_dict = metadata_dict["prompt"]
    assert isinstance(prompt_dict, dict)
    return dict(
        condition_type=prompt_dict["condition_type"],
        actual_genre=prompt_dict["track_fpath"].split("_")[0],
        track_fpath=prompt_dict["track_fpath"]
    )


def format_rating(response_dict: dict) -> dict:
    """Formats JSON object for `question`: `rating`"""
    # Grab the metadata and convert to a dictionary
    metadata = ast.literal_eval(response_dict["metadata_"])
    metadata_fmt = format_metadata(metadata)
    # Grab the answer dictionary and format
    answer = ast.literal_eval(response_dict["answer"])
    answer_fmt = format_answer(answer)
    # Get some extra information
    extras = {
        "correct": answer_fmt["predicted_genre"] == metadata_fmt["actual_genre"],  # if predicted genre == actual genre
        "response_id": coerce_type(response_dict["id"]),
        "participant_id": coerce_type(response_dict["participant_id"]),
        "time_taken": coerce_type(metadata["time_taken"])
    }
    # Return as a single, big, dictionary
    return extras | metadata_fmt | answer_fmt


def main(export_json: str = EXPORT_JSON):
    # Load in the JSON: this will nicely raise an error if the file can't be found
    read_json = utils.read_json_cached(export_json)
    # Variables to hold different types of responses
    all_answers = []
    participant_numeric_metadatas = {nt: [] for nt in METADATA_NUMERIC_TYPES}
    participant_categoric_metadatas = {ct: Counter() for ct in METADATA_CATEGORICAL_TYPES}
    participant_feedback_metadatas = {nt: [] for nt in METADATA_FEEDBACK_TYPES}
    participant_emails = []

    # Iterate over every dictionary inside the list
    for response in read_json:
        # Formatting RATINGS: these are our responses to each triplet of stimuli
        if response["question"] == "rating":
            # Throw a warning if we've somehow failed here
            if response["failed"] != "false":
                logger.error(f"Found failed response, participant id {response['participant_id']}: "
                             f"reason {response['failed_reason']}")
                continue
            all_answers.append(format_rating(response))

        # Formatting demographic questions with numeric type, e.g. age, hours of music listening, etc.
        elif response["question"] in METADATA_NUMERIC_TYPES:
            answer = coerce_type(response["answer"])
            if response["question"] == "hours_of_daily_music_listening" and answer > 20:
                response["answer"] = None  # again, a fucking undergrad thought it'd be funny to put 24
            participant_numeric_metadatas[response["question"]].append(answer)

        # Formatting demographic questions with categorical type
        elif response["question"] in METADATA_CATEGORICAL_TYPES:
            if response["answer"] not in ["male", "female", "non-binary"] and response["question"] == "gender":
                response["answer"] = "male"  # this is because some fucking undergrad put "Drummer"
            participant_categoric_metadatas[response["question"]][response["answer"]] += 1

        # Formatting demographic questions with string type
        elif response["question"] in METADATA_FEEDBACK_TYPES:
            participant_feedback_metadatas[response["question"]].append(response["answer"])

        # Formatting participant emails
        elif response["question"] == "email":
            if response["answer"] is None:
                pass
            # Simple way of filtering if somebody actually provided an email
            elif "@" in response["answer"]:
                participant_emails.append(response["answer"])
            else:
                logger.warning(f"Email obtained, but no @ symbol found: {response['answer']}")

    # Convert list of dictionaries to a DataFrame
    answers_df = pd.DataFrame(all_answers)

    logger.info("------RESPONSES------")
    # Grab the number of participants and responses from the answer list
    n_participants = len(set(a["participant_id"] for a in all_answers))
    n_responses = len(all_answers)
    logger.debug(f"Obtained {n_participants} participant(s) with {n_responses} total ratings")
    # Organise the number of responses according to condition
    grped_by_condition = answers_df.groupby("condition_type")["response_id"].size()
    logger.debug(f"Ratings per condition: {grped_by_condition.mean():.3f}")

    # Create plots for similarity judgement
    logger.info("------SIMILARITY QUESTION------")
    heatmap = plotting.HeatmapSubjectiveAccuracy(answers_df)
    heatmap.create_plot()
    heatmap.save_fig(os.path.join(FIGURES_DIR, "heatmap_accuracy"))
    for idx, ct in answers_df.groupby("condition_type"):
        accuracy = (len(ct[ct["correct"] == True]) / len(ct)) * 100
        logger.info(f"Condition {idx}, accuracy {accuracy:.3f}%")

    # Create plots for quality judgement (Likert scales)
    logger.info("------QUALITY QUESTIONS------")
    quality_bp = plotting.BarPlotSubjectiveQuality(answers_df)
    quality_bp.create_plot()
    quality_bp.save_fig(os.path.join(FIGURES_DIR, "barplot_quality"))

    logger.info("------DEMOGRAPHIC QUESTIONS------")

    # Log average and standard deviation for all numeric types
    for numeric_type, numeric_vals in participant_numeric_metadatas.items():
        numeric_vals = np.array(numeric_vals, dtype=float)
        logger.debug(f"Question {numeric_type}: mean {np.nanmean(numeric_vals):.3f}, SD {np.nanstd(numeric_vals):.3f}")

    # Log counts and proportions for all categorical types
    for numeric_type, numeric_vals in participant_categoric_metadatas.items():
        counts = ", ".join(f"{k}, {v} ({((v / numeric_vals.total()) * 100):.3f} %)" for k, v in numeric_vals.items())
        logger.debug(f"Question {numeric_type}: {counts}")

    # Do the prize draw
    logger.info("------PRIZE DRAW------")
    if len(participant_emails) > 0:
        lucky_winner = random.choice(participant_emails)
    else:
        lucky_winner = "Nobody! Save your money, Huw!!"
    logger.debug(f"The winner of the £50 prize draw is: {lucky_winner}")


if __name__ == "__main__":
    utils.seed_everything(utils.SEED)
    main()
