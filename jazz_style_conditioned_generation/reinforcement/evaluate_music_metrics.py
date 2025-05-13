#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluates generated MIDI files using notes-per-second and sliding pitch-class-entropy"""

import os

import numpy as np
from loguru import logger
from tqdm import tqdm

from jazz_style_conditioned_generation import utils
from jazz_style_conditioned_generation.data import DATA_DIR
from jazz_style_conditioned_generation.data.scores import load_score, preprocess_score, get_notes_from_score
from jazz_style_conditioned_generation.metrics import sliding_event_density, sliding_pitch_class_entropy
from jazz_style_conditioned_generation.preprocessing.splits import SPLIT_DIR

JAZZ_DATA_DIR = os.path.join(DATA_DIR, "raw")
GENERATION_DIR = os.path.join(utils.get_project_root(), "data/rl_generations")


def read_tracks_for_splits(split_type: str) -> list[str]:
    """Given the name of a split (train, test, validation), get corresponding filepaths"""
    split_fp = os.path.join(SPLIT_DIR, f"{split_type}_split.txt")
    tracks = []
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
            tracks.append(track_path)
    utils.validate_paths(tracks, expected_extension=".mid")
    return tracks


def get_generations(generation_path: str, generation_iter: int) -> list[str]:
    """Gets all generations from the current iteration from generation_path"""
    tracks = []
    generation_full_path = os.path.join(GENERATION_DIR, generation_path)
    # Iterate over all files inside our generated path
    for i in os.listdir(generation_full_path):
        if all((
                int(i.split("_")[1].replace("iter", "")) == generation_iter,  # should be correct iteration
                i.endswith(".mid"),  # should be a MIDI file
        )):
            full_path = os.path.join(generation_full_path, i)
            tracks.append(full_path)
    utils.validate_paths(tracks, expected_extension=".mid")
    return tracks


def compute_metrics(track_path: str, preprocess: bool = True) -> tuple[float, float]:
    """Given a score, calculate NPS and PCE"""
    try:
        score = load_score(track_path, as_seconds=True)
        # Preprocess score if required (only for ground-truth tracks, generated tracks are already preprocessed)
        if preprocess:
            score = preprocess_score(score)
        notes = get_notes_from_score(score)  # converts symusic.Score = [symusic.Note, symusic.Note, ...]
    except IndexError:
        nps, pce = None, None
        # logger.warning(f"Errored for track {track_path}: {e}")
    else:
        nps = sliding_event_density(notes)
        pce = sliding_pitch_class_entropy(notes)
    return nps, pce


def main(generation_path: str, generation_iter: int):
    # Grab filepaths to test tracks and generated tracks
    test_tracks = read_tracks_for_splits("test")
    generation_tracks = get_generations(generation_path, generation_iter)

    # Iterate over all tracks in the test dataset
    test_pces, test_nps = [], []
    for test_track in tqdm(test_tracks, desc="Computing metrics for held-out test tracks"):
        # Need to preprocess the ground-truth tracks
        nps, pce = compute_metrics(test_track, preprocess=True)
        # Append everything to our lists
        if nps is not None:
            test_nps.append(nps)
        if pce is not None:
            test_pces.append(pce)
    # Log everything
    mean_test_pce, mean_test_nps = np.nanmean(test_pces), np.nanmean(test_nps)
    logger.debug(f"Mean test sliding PCE {mean_test_pce:.3f}, NPS {mean_test_nps:.3f}")

    # Iterate over all generations
    gen_pces, gen_nps = [], []
    for gen_track in tqdm(generation_tracks, desc=f"Computing metrics for generation iter {generation_iter}"):
        # No need to preprocess generated MIDI
        nps, pce = compute_metrics(gen_track, preprocess=False)
        # Append everything to our lists
        if nps is not None:
            gen_nps.append(nps)
        if pce is not None:
            gen_pces.append(pce)
    # Log everything
    mean_gen_pce, mean_gen_nps = np.nanmean(gen_pces), np.nanmean(gen_nps)
    logger.debug(f"Mean generated sliding PCE {mean_gen_pce:.3f}, NPS {mean_gen_nps:.3f}, iteration {generation_iter}")


if __name__ == "__main__":
    import argparse

    # Seed everything for reproducible results
    utils.seed_everything(utils.SEED)
    # Parsing arguments from the command line interface
    parser = argparse.ArgumentParser(description="Evaluate sliding PCE and NPS of generated MIDI files")
    parser.add_argument(
        "-d", "--generation-dir", type=str, help="Path to generated MIDIs",
        default="finetuning-customtok-plateau/finetuning_customtok_10msmin_lineartime_"
                "moreaugment_init6e5reduce10patience5_batch4_1024seq_12l8h768d3072ff"
    )
    parser.add_argument(
        "-i", "--generation-iter", type=int, help="Iteration of generations to use.", default=1
    )
    args = vars(parser.parse_args())
    # Run everything
    main(
        generation_path=args["generation_dir"],
        generation_iter=args["generation_iter"],
    )
