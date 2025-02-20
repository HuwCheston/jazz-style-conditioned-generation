#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Create data splits"""

import os
from itertools import combinations

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from jazz_style_conditioned_generation import utils

TRAIN_SIZE, TEST_SIZE, VALIDATION_SIZE = 0.8, 0.1, 0.1
SPLIT_TYPES = ["train", "test", "validation"]

DATA_ROOT = os.path.join(utils.get_project_root(), 'data/raw/')
SPLIT_DIR = os.path.join(utils.get_project_root(), "references/data_splits")


def get_tracks(midi_fname: str = "piano_midi.mid"):
    """Gets paths to folders for all tracks inside data root directory"""
    for (root, subdir, files) in os.walk(DATA_ROOT):
        if midi_fname in files:
            # this just gives us "<DATABASE>/<TRACK_NAME>
            yield root.replace(DATA_ROOT, "")


def get_tracks_in_existing_split(split_name: str) -> list[str]:
    """Returns the names of the tracks from an existing data split"""
    split_path = os.path.join(SPLIT_DIR, "from_performer_identification_work", split_name + "_split.csv")
    df = pd.read_csv(split_path)
    return df["track"].values.tolist()


def get_pianist(fname: str) -> str:
    """Loads metadata for a given track and returns the name of the pianist"""
    js_path = os.path.join(DATA_ROOT, fname, "metadata_tivo.json")
    return utils.read_json_cached(js_path)["pianist"]


def check_all_splits_unique(*splits: list[str]) -> None:
    """Takes in an arbitrary number of data splits and checks that they are all unique"""
    for sp1, sp2 in combinations(splits, 2):
        assert not any(i in sp2 for i in sp1), "Overlap in data splits!"


def dump_split_txt_file(split: list[str], split_name: str) -> None:
    """Dumps the tracks within a split inside a text file"""
    split_fpath = os.path.join(SPLIT_DIR, f'{split_name}_split.txt')
    with open(split_fpath, 'w') as f:
        for line in split:
            f.write(f"{line}\n")


def main():
    all_tracks = list(get_tracks())
    starting_track_count = len(all_tracks)
    logger.info(f"Found {len(all_tracks)} tracks inside {DATA_ROOT}")
    splits = {ty: [] for ty in SPLIT_TYPES}

    # We want to reuse the splits from our previous work
    for split in SPLIT_TYPES:
        existing_tracks = get_tracks_in_existing_split(split)
        splits[split].extend(existing_tracks)
        all_tracks = [t for t in all_tracks if t not in existing_tracks]
    logger.info(f"{starting_track_count - len(all_tracks)} tracks used from performer identification splits")

    # Convert everything to numpy arrays
    tracks = np.array(all_tracks)
    datasets = np.array([0 if "jtd/" in t else 1 for t in tracks])
    pianists = np.array([get_pianist(t) for t in tracks])
    assert len(tracks) == len(datasets) == len(pianists)

    # Split into train/held-out sets (stratified by source dataset)
    train_tracks, temp_tracks, train_pianist, temp_pianist = train_test_split(
        tracks,
        pianists,
        test_size=TEST_SIZE + VALIDATION_SIZE,
        train_size=TRAIN_SIZE,
        stratify=datasets,
        random_state=utils.SEED,
        shuffle=True
    )
    splits["train"].extend(train_tracks)

    # Split held out set into validation and test sets
    v_size = VALIDATION_SIZE / (TEST_SIZE + VALIDATION_SIZE)
    t_size = TEST_SIZE / (TEST_SIZE + VALIDATION_SIZE)
    assert round(v_size + t_size) == 1.
    temp_databases = np.array([0 if "jtd/" in t else 1 for t in temp_tracks])
    valid_tracks, test_tracks, valid_pianist, test_pianist = train_test_split(
        temp_tracks,
        temp_pianist,
        stratify=temp_databases,
        test_size=t_size,
        train_size=v_size,  # Use validation as train
        random_state=utils.SEED,
        shuffle=True
    )
    splits["test"].extend(test_tracks)
    splits["validation"].extend(valid_tracks)

    # Confirm that there's no overlap in the splits
    check_all_splits_unique(splits["train"], splits["validation"], splits["test"])
    # Confirm that we have the same number of tracks that we started with
    assert sum([len(sp) for sp in splits.values()]) == starting_track_count
    # Dump the tracks for this split as a text file
    for split_name, split_tracks in splits.items():
        logger.info(f"{len(split_tracks)} tracks in {split_name} split")
        dump_split_txt_file(split_tracks, split_name)
    logger.info(f"All splits created and dumped inside {SPLIT_DIR}!")
    # MIDI files will be chunked in the training module as they depend on a particular tokenizer


if __name__ == "__main__":
    utils.seed_everything(utils.SEED)
    main()
