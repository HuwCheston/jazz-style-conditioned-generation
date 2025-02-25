#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Add human-confirmed blindfold test descriptions to metadata for the corresponding tracks"""

import os

import pandas as pd
from loguru import logger

from jazz_style_conditioned_generation import utils

# This is the filepath towards the CSV file containing the human-confirmed matches
CONFIRMED_MATCH_FILEPATH = os.path.join(utils.get_project_root(), "outputs/blindfold_tests/confirmed_matches.csv")
# Unique directories for all tracks in the dataset: this little hack gets the directory name by globbing JSONs
TRACK_DIRS = list(set([os.path.dirname(i) for i in utils.get_data_files_with_ext("data/raw", "**/*.json")]))


def main():
    # Load in the dataframe and subset to get only the human-confirmed matches
    df = pd.read_csv(CONFIRMED_MATCH_FILEPATH)
    confirmed_matches = df[df["KEEP"] == "Y"]
    # Exact matches: these are the FILEPATHS to the tracks that exactly match a track referenced in downbeat
    exact_matches = confirmed_matches["track_fp"].unique()
    # Album/performer matches: these are the names of performers/albums that have 1+ track referenced in downbeat
    album_matches = confirmed_matches["album_name"].unique()
    performer_matches = confirmed_matches["performer_name"].unique()
    # Counters
    exacts, nears, misses = 0, 0, 0
    # Iterate over the directory for every track in the dataset
    for track_dir in TRACK_DIRS:
        # Load in the metadata JSON for this track
        metadata_fp = os.path.join(track_dir, "metadata.json")
        assert os.path.isfile(metadata_fp)
        loaded_metadata = utils.read_json_cached(metadata_fp)
        # Add a new empty list to the dictionary
        loaded_metadata["blindfold_responses"] = []
        # Track is an exact match (i.e., this EXACT recording is referred to in downbeat)
        if loaded_metadata["fname"] in exact_matches:
            exacts += 1
            match = confirmed_matches[confirmed_matches["track_fp"] == loaded_metadata["fname"]]
            for idx_, response in match.iterrows():
                loaded_metadata["blindfold_responses"].append(dict(
                    response=response["section_of_text"],
                    blindfold_test_filename=os.path.basename(response["blindfold_fp"]),
                    match_type="exact"
                ))
        # Track is a near match: it is from the same album as another track which is referred to in downbeat
        elif (loaded_metadata["pianist"] in performer_matches and
              loaded_metadata["album_name"] in album_matches):
            nears += 1
            match = confirmed_matches[
                (confirmed_matches["performer_name"] == loaded_metadata["pianist"]) &
                (confirmed_matches["album_name"] == loaded_metadata["album_name"])
                ]
            for idx_, response in match.iterrows():
                loaded_metadata["blindfold_responses"].append(dict(
                    response=response["section_of_text"],
                    blindfold_test_filename=os.path.basename(response["blindfold_fp"]),
                    match_type="near"
                ))
        # Track is a miss
        else:
            misses += 1
        # Dump the JSON to a new file in the root directory

    # Sanity check and logging
    assert sum((exacts, nears, misses)) == len(TRACK_DIRS)
    logger.info(f"Total number of tracks: {len(TRACK_DIRS)}")
    for match_type, matches in zip(["exact", "near", "miss"], [exacts, nears, misses]):
        logger.info(f'Number of {match_type} matches: {matches} ({(matches / len(TRACK_DIRS)) * 100:.3f}%)')


if __name__ == '__main__':
    main()
