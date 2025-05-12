#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluates generated MIDI files using label (pianist/genre) accuracy"""

import os
import pickle

import numpy as np
import torch
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from jazz_style_conditioned_generation import utils
from jazz_style_conditioned_generation.data import DATA_DIR
from jazz_style_conditioned_generation.data.scores import load_score, preprocess_score
from jazz_style_conditioned_generation.data.tokenizer import (
    load_tokenizer,
    add_pianists_to_vocab,
    add_tempos_to_vocab,
    add_timesignatures_to_vocab,
    add_recording_years_to_vocab,
    add_genres_to_vocab
)
from jazz_style_conditioned_generation.preprocessing.splits import SPLIT_DIR, check_all_splits_unique
from jazz_style_conditioned_generation.reinforcement import clamp_utils

JAZZ_DATA_DIR = os.path.join(DATA_DIR, "raw")
GENERATION_DIR = os.path.join(utils.get_project_root(), "data/rl_generations")

MAX_SEQ_LEN = 1024

# Initialise the tokenizer, add all the condition tokens in
TOKENIZER = load_tokenizer(tokenizer_str="custom-tsd", tokenizer_kws=dict(time_range=[0.01, 1.0], time_factor=1.0))
add_genres_to_vocab(TOKENIZER)
add_pianists_to_vocab(TOKENIZER)
add_recording_years_to_vocab(TOKENIZER, 1945, 2025, step=5)  # [1945, 1950, ..., 2025]
add_tempos_to_vocab(TOKENIZER, 80, 30, factor=1.05)
add_timesignatures_to_vocab(TOKENIZER, [3, 4])

CLASSIFIER_FPATH = os.path.join(utils.get_project_root(), "references/label_accuracy_classifier.p")

CLAMP = clamp_utils.initialize_clamp(pretrained=True)

PIANISTS = ["Bill Evans", "Oscar Peterson", "Ahmad Jamal", "Keith Jarrett", "McCoy Tyner"]
PIANIST_MAPPING = {p: n for n, p in enumerate(sorted(PIANISTS))}
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


def extract_ground_truth_features(tracks: list[str], metas: list[str] | list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Given a list of track and metadata filepaths corresponding to real tracks, extract features + target variables"""
    xs, ys = [], []
    for train_track, train_meta in tqdm(zip(tracks, metas), total=len(tracks), desc="Extracting features..."):
        # If we haven't loaded the JSON already, do this now
        if isinstance(train_meta, str):
            train_meta = utils.read_json_cached(train_meta)
        pianist = train_meta["pianist"]
        # If the track is by one of our 25 pianists
        if pianist in PIANIST_MAPPING.keys():
            # Load up the score and preprocess it
            loaded = load_score(train_track, as_seconds=True)
            loaded = preprocess_score(loaded)
            # Encode the score with our tokenizer and get the IDs
            track_encoded = torch.tensor([TOKENIZER(loaded)[0].ids])
            # Iterate over 1024-token chunks
            for seq_start in range(0, track_encoded.size(1), MAX_SEQ_LEN):
                # Chunk the token input
                seq_end = seq_start + MAX_SEQ_LEN
                ids_chunked = track_encoded[:, seq_start:seq_end]
                # Convert to expected clamp input
                chunk_clamp_input = clamp_utils.midi_to_clamp(ids_chunked, TOKENIZER)
                # Extract features with clamp
                chunk_clamp_features = clamp_utils.extract_clamp_features(chunk_clamp_input, CLAMP, get_global=True)
                # Append extracted features and pianist IDX to the list
                xs.append(chunk_clamp_features.cpu())
                ys.append(PIANIST_MAPPING[pianist])
            # Cleanup if required
            if os.path.exists("tmp.mid"):
                os.remove("tmp.mid")
    # Stack xs to (N_tracks, N_dims), ys to (N_tracks)
    return np.stack(xs), np.stack(ys)


def extract_generated_features(tracks: list[str], metas: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Given a list of track and metadata filepaths corresponding to generations, extract features + target variables"""
    xs, ys = [], []
    for train_track, train_meta in tqdm(zip(tracks, metas), total=len(tracks), desc="Extracting features..."):
        # If we haven't loaded the JSON already, do this now
        if isinstance(train_meta, str):
            train_meta = utils.read_json_cached(train_meta)
        pianist = train_meta["pianist"]
        # If the track is by one of our 25 pianists
        if pianist in PIANIST_MAPPING.keys():
            # No need to load score + convert to tokens, generations will always be 1024 tokens long
            # Convert to expected clamp input
            chunk_clamp_input = clamp_utils.midi_to_clamp(train_track)
            # Extract features with clamp
            chunk_clamp_features = clamp_utils.extract_clamp_features(chunk_clamp_input, CLAMP, get_global=True)
            # Append extracted features and pianist IDX to the list
            xs.append(chunk_clamp_features.cpu())
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


def main(generation_path: str, generation_iter: int = 0, force_training: bool = False):
    # Load up training + test track + metadata paths
    train_tracks, train_metas = read_tracks_for_splits("train")
    test_tracks, test_metas = read_tracks_for_splits("test")
    logger.debug(f"Loaded {len(train_tracks)} training tracks, {len(test_tracks)} testing tracks")
    # Load up artificially generated tracks
    generation_path = os.path.join(GENERATION_DIR, generation_path)
    generated_tracks, generated_metas = get_generations(generation_path, generation_iter)
    logger.debug(f"Loaded {len(generated_tracks)} generations")
    # Get features for tracks by all pianists
    train_xs, train_ys = extract_ground_truth_features(train_tracks, train_metas)
    logger.debug(f"Extracted training features with CLaMP-3: x shape {train_xs.shape}, y shape {train_ys.shape}")
    test_xs, test_ys = extract_ground_truth_features(test_tracks, test_metas)
    logger.debug(f"Extracted testing features with CLaMP-3: x shape {test_xs.shape}, y shape {test_ys.shape}")
    # Get features for all generated tracks
    gen_xs, gen_ys = extract_generated_features(generated_tracks, generated_metas)
    logger.debug(f"Extracted generated features with CLaMP-3: x shape {gen_xs.shape}, y shape {gen_ys.shape}")
    # Scale the data
    scaler = StandardScaler()
    all_xs = np.concatenate([train_xs, test_xs, gen_xs], axis=0)
    scaler.fit(all_xs)
    train_xs = scaler.transform(train_xs)
    test_xs = scaler.transform(test_xs)
    gen_xs = scaler.transform(gen_xs)
    # Initialize the model
    logger.debug("Fitting model...")
    # Try and load a fitted model from disk
    if os.path.exists(CLASSIFIER_FPATH) and not force_training:
        with open(CLASSIFIER_FPATH, "rb") as f:
            model = pickle.load(f)
        logger.debug(f"... loaded model from {CLASSIFIER_FPATH}")
    # Train the model from scratch
    else:
        model = LogisticRegression(random_state=utils.SEED, penalty="l2", max_iter=10000)
        model.fit(train_xs, train_ys)
        # Dump to the disk
        with open(CLASSIFIER_FPATH, "wb") as f:
            pickle.dump(model, f, protocol=5)
        logger.debug(f"... model dumped to {CLASSIFIER_FPATH}")
    # Predict the real-world (test) data
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
    parser.add_argument(
        "-f", "--force-training", type=utils.string_to_bool, help="Force retraing", default=False
    )
    args = vars(parser.parse_args())
    # Run everything
    main(
        generation_path=args["generation_dir"],
        generation_iter=args["generation_iter"],
        force_training=args["force_training"]
    )
