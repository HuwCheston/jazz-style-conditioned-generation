#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluates generated MIDI files using label (pianist/genre) accuracy"""

import os

import numpy as np
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from jazz_style_conditioned_generation import utils
from jazz_style_conditioned_generation.data import DATA_DIR
from jazz_style_conditioned_generation.data.conditions import INCLUDE
from jazz_style_conditioned_generation.preprocessing.splits import SPLIT_DIR, check_all_splits_unique
from jazz_style_conditioned_generation.reinforcement import clamp_utils

JAZZ_DATA_DIR = os.path.join(DATA_DIR, "raw")
GENERATION_DIR = os.path.join(utils.get_project_root(), "data/rl_generations")

CLAMP = clamp_utils.initialize_clamp(pretrained=True)

PIANIST_MAPPING = {p: n for n, p in enumerate(sorted(INCLUDE["pianist"]))}
PIANIST_FMT = {utils.remove_punctuation(p).lower().replace(" ", ""): p for p in PIANIST_MAPPING}


def read_tracks_for_splits(split_type: str):
    """Given the name of a split (train, test, validation), get corresponding filepaths"""
    split_fp = os.path.join(SPLIT_DIR, f"{split_type}_split.txt")
    tracks, metadatas = [], []
    with open(split_fp, 'r') as fp:
        all_paths = fp.read().strip().split('\n')
        # Check that the path exists on the local file structure
        for path in all_paths:
            # Skip over empty lines
            if path == "":
                continue
            track_path = os.path.join(JAZZ_DATA_DIR, path, "piano_midi.mid")
            if not os.path.isfile(track_path):
                raise FileNotFoundError(f'Could not find MIDI for track at {track_path}')
            metadata_path = os.path.join(JAZZ_DATA_DIR, path, "metadata_tivo.json")
            if not os.path.isfile(metadata_path):
                raise FileNotFoundError(f'Could not find metadata for track at {metadata_path}')
            tracks.append(track_path)
            metadatas.append(metadata_path)
    validate_splits(tracks, metadatas)
    return tracks, metadatas


def validate_splits(track_splits: dict[str, str], metadata_splits: dict[str, str]) -> None:
    """Validates track + metadata splits are unique and present on disk"""
    check_all_splits_unique(track_splits)
    check_all_splits_unique(metadata_splits)
    utils.validate_paths(track_splits, expected_extension=".mid")
    utils.validate_paths(metadata_splits, expected_extension=".json")


def extract_features(tracks: list[str], metas: list[str] | list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Given a list of track and metadata filepaths, extract features + target variables"""
    xs, ys = [], []
    for train_track, train_meta in tqdm(zip(tracks, metas), total=len(tracks), desc="Extracting features..."):
        # If we haven't loaded the JSON already, do this now
        if isinstance(train_meta, str):
            train_meta = utils.read_json_cached(train_meta)
        pianist = train_meta["pianist"]
        # If the track is by one of our 25 pianists
        if pianist in PIANIST_MAPPING.keys():
            # Extract the features and append to the list
            track_clamp_input = clamp_utils.midi_to_clamp(train_track)
            track_clamp_features = clamp_utils.extract_clamp_features(track_clamp_input, CLAMP)
            # Append extracted features and pianist IDX to the list
            xs.append(track_clamp_features.cpu())  # should be on CPU for sklearn + numpy
            ys.append(PIANIST_MAPPING[pianist])
    # Stack xs to (N_tracks, N_dims), ys to (N_tracks)
    return np.stack(xs), np.stack(ys)


def get_generations(generation_path: str, generation_iter: int) -> tuple[list[str], dict[str, str]]:
    """Gets all generations from the current iteration from generation_path"""
    tracks, metas = [], []
    # Iterate over all files inside our generated path
    for i in os.listdir(generation_path):
        if all((
                int(i.split("_")[1].replace("iter", "")) == generation_iter,  # should be correct iteration
                i.endswith(".mid"),  # should be a MIDI file
                i.split("_")[0] in PIANIST_FMT.keys()  # should be by one of our pianists
        )):
            full_path = os.path.join(generation_path, i)
            utils.validate_paths([full_path], expected_extension=".mid")
            fake_meta = {"pianist": PIANIST_FMT[i.split("_")[0]]}
            tracks.append(full_path)
            metas.append(fake_meta)
    return tracks, metas


def main(generation_path: str, generation_iter: int = 0):
    # Load up training + test track + metadata paths
    train_tracks, train_metas = read_tracks_for_splits("train")
    test_tracks, test_metas = read_tracks_for_splits("test")
    logger.debug(f"Loaded {len(train_tracks)} training tracks, {len(test_tracks)} testing tracks")
    # Load up artificially generated tracks
    generation_path = os.path.join(GENERATION_DIR, generation_path)
    generated_tracks, generated_metas = get_generations(generation_path, generation_iter)
    logger.debug(f"Loaded {len(generated_tracks)} generations")
    # Get features for tracks by all pianists
    train_xs, train_ys = extract_features(train_tracks, train_metas)
    logger.debug(f"Extracted training features with CLaMP-3: x shape {train_xs.shape}, y shape {train_ys.shape}")
    test_xs, test_ys = extract_features(test_tracks, test_metas)
    logger.debug(f"Extracted testing features with CLaMP-3: x shape {test_xs.shape}, y shape {test_ys.shape}")
    # Get features for all generated tracks
    gen_xs, gen_ys = extract_features(generated_tracks, generated_metas)
    # Scale the data
    scaler = StandardScaler()
    train_xs = scaler.fit_transform(train_xs)
    test_xs = scaler.transform(test_xs)
    gen_xs = scaler.transform(gen_xs)
    # Initialize the model
    logger.debug("Fitting model...")
    model = LogisticRegression(random_state=utils.SEED, penalty="l2")
    # Fit the model and predict the (real) test data
    model.fit(train_xs, train_ys)
    test_acc = model.score(test_xs, test_ys)
    logger.debug(f"... accuracy predicting real-world test data: {test_acc:.3f}")
    # Predict the generated data
    gen_acc = model.score(gen_xs, gen_ys)
    logger.debug(f"... accuracy predicting generated data: {gen_acc:.3f}")


if __name__ == "__main__":
    import argparse

    # Seed everything for reproducible results
    utils.seed_everything(utils.SEED)
    # Parsing arguments from the command line interface
    parser = argparse.ArgumentParser(description="Evaluate label accuracy of generated MIDI files")
    parser.add_argument(
        "-d", "--generation-dir", type=str, help="Path to generated MIDIs",
        default="finetuning-customtok-plateau/finetuning_customtok_10msmin_lineartime_"
                "moreaugment_init6e5reduce10patience5_batch4_1024seq_12l8h768d3072ff"
    )
    parser.add_argument(
        "-i", "--generation-iter", type=int, help="Iteration of generations to use.", default=0
    )
    args = vars(parser.parse_args())
    # Run everything
    main(generation_path=args["generation_dir"], generation_iter=args["generation_iter"])
