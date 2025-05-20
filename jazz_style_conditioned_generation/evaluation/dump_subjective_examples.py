#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Dumps MIDI examples for subjective listening test"""

import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from jazz_style_conditioned_generation import utils
from jazz_style_conditioned_generation.data.scores import load_score, preprocess_score
from jazz_style_conditioned_generation.data.tokenizer import (
    load_tokenizer,
    add_pianists_to_vocab,
    add_tempos_to_vocab,
    add_timesignatures_to_vocab,
    add_recording_years_to_vocab,
    add_genres_to_vocab
)
from jazz_style_conditioned_generation.evaluation.evaluate_music_metrics import read_tracks_for_splits
from jazz_style_conditioned_generation.reinforcement import clamp_utils
from jazz_style_conditioned_generation.reinforcement.rl_train import GroundTruthDataset

CONDITION_TOKENS = [
    "Avant-Garde Jazz",
    "Straight-Ahead Jazz",
    "Traditional & Early Jazz",
    "Global",
    "Soul Jazz"
]
TRAIN_TRACKS = read_tracks_for_splits("train")
TEST_TRACKS = read_tracks_for_splits("test")
VALID_TRACKS = read_tracks_for_splits("validation")
HELD_OUT_TRACKS = TEST_TRACKS + VALID_TRACKS

N_TRACKS = 10
MAX_SEQ_LEN = 1024

# Initialise the tokenizer, add all the condition tokens in
TOKENIZER = load_tokenizer(tokenizer_str="custom-tsd", tokenizer_kws=dict(time_range=[0.01, 1.0], time_factor=1.0))
add_genres_to_vocab(TOKENIZER)
add_pianists_to_vocab(TOKENIZER)
add_recording_years_to_vocab(TOKENIZER, 1945, 2025, step=5)  # [1945, 1950, ..., 2025]
add_tempos_to_vocab(TOKENIZER, 80, 30, factor=1.05)
add_timesignatures_to_vocab(TOKENIZER, [3, 4])

EXAMPLES_DIR = os.path.join(utils.get_project_root(), "outputs/listening_test_examples")
if not os.path.isdir(EXAMPLES_DIR):
    os.mkdir(EXAMPLES_DIR)


class TestDataset(GroundTruthDataset):
    """Custom dataset that also returns track filepaths"""

    def __getitem__(self, item: int):
        # Return both the filepath and the extracted features
        fpath = self.files_paths[item]
        features = super().__getitem__(item)
        return fpath, features


class GenerationDataset(TestDataset):
    def get_tracks_with_condition(self, files_paths: list[str]):
        tok_fmt = utils.remove_punctuation(self.condition_token).replace(' ', '').lower()
        for track in tqdm(
                files_paths,
                total=len(files_paths),
                desc=f"Getting generated tracks with condition {self.condition_token}"
        ):
            if track.split(os.path.sep)[-1].startswith(tok_fmt) and track.endswith(".mid"):
                yield track


def extract_features_and_sort(dataset: torch.utils.data.Dataset, gt_features: torch.Tensor) -> list[str]:
    test_res = []
    for i in tqdm(range(len(dataset)), "Loading examples..."):
        fpath, features = dataset[i]
        sim = F.cosine_similarity(features, gt_features, dim=-1).mean().item()
        test_res.append((fpath, sim))
    return [i for i, _ in sorted(test_res, key=lambda x: x[1], reverse=True)][:N_TRACKS]


def process_ground_truth_track(filepath: str):
    # Load and preprocess score in seconds
    loaded = preprocess_score(load_score(filepath, as_seconds=True))
    # Encode as tokens
    encoded = TOKENIZER(loaded)[0].ids
    # Get a random starting point for the 1024-token chunk
    #  Should be "within bounds" for the track
    start_point = np.random.randint(0, max(len(encoded) - MAX_SEQ_LEN, 0))
    end_point = start_point + MAX_SEQ_LEN
    chunk = encoded[start_point:end_point]
    # Back to a Score object to let us dump it to disk
    return TOKENIZER(torch.tensor([chunk]))


def main(generation_dir_clamp: str, generation_dir_noclamp: str) -> None:
    clamp = clamp_utils.initialize_clamp(pretrained=True)
    generation_dir_clamp = os.path.join(utils.get_project_root(), "data/rl_generations", generation_dir_clamp)
    generation_dir_noclamp = os.path.join(utils.get_project_root(), "data/rl_generations", generation_dir_noclamp)

    # Load up generations made USING CLAMP
    gen_fps_clamp = [
        os.path.join(generation_dir_clamp, i) for i in os.listdir(generation_dir_clamp)
        if i.endswith(".mid")  # should be a MIDI file
           and "iter003" in i  # should be generated with k=3 iterations of CLaMP-DPO loss
    ]
    utils.validate_paths(gen_fps_clamp, expected_extension=".mid")
    # Load up generations made WITHOUT CLAMP
    gen_fps_noclamp = [
        os.path.join(generation_dir_noclamp, i) for i in os.listdir(generation_dir_noclamp)
        if i.endswith(".mid")  # should be a MIDI file
           and "iter000" in i  # should be generated with k=0 iterations of CLaMP-DPO loss
    ]
    utils.validate_paths(gen_fps_noclamp, expected_extension=".mid")

    for condition_token in CONDITION_TOKENS:
        tok_fmt = utils.remove_punctuation(condition_token).replace(' ', '').lower()
        logger.info(f"Processing token {tok_fmt}")
        # Get ground truth tracks from the training dataset with this condition
        train_dataset = DataLoader(
            GroundTruthDataset(
                files_paths=TRAIN_TRACKS,
                condition_token=condition_token,
                clamp=clamp
            ),
            shuffle=False,
            drop_last=False,
            batch_size=4
        )
        gt_features = torch.cat([i for i in tqdm(train_dataset, desc="Loading train examples...")], dim=0)

        generated_dataset_clamp = GenerationDataset(files_paths=gen_fps_clamp, condition_token=condition_token,
                                                    clamp=clamp)
        # Compute cosine similarities and store filenames + sims
        gen_res_sorted_clamp = extract_features_and_sort(generated_dataset_clamp, gt_features)
        # Iterate over the top N most similar generations
        for n, t in enumerate(gen_res_sorted_clamp):
            # No need to do anything special here, we can just copy the generation with shutil
            dst = os.path.join(EXAMPLES_DIR, f"{tok_fmt}_{str(n).zfill(3)}_gen_clamp.mid")
            shutil.copy(t, dst)

        generated_dataset_noclamp = GenerationDataset(
            files_paths=gen_fps_noclamp,
            condition_token=condition_token,
            clamp=clamp
        )
        # Compute cosine similarities and store filenames + sims
        gen_res_sorted_noclamp = extract_features_and_sort(generated_dataset_noclamp, gt_features)
        # Iterate over the top N most similar generations
        for n, t in enumerate(gen_res_sorted_noclamp):
            # No need to do anything special here, we can just copy the generation with shutil
            dst = os.path.join(EXAMPLES_DIR, f"{tok_fmt}_{str(n).zfill(3)}_gen_noclamp.mid")
            shutil.copy(t, dst)

        # Do the same for the held-out test data (both validation and test tracks)
        test_dataset = TestDataset(
            files_paths=HELD_OUT_TRACKS,
            condition_token=condition_token,
            clamp=clamp
        )
        test_res_sorted = extract_features_and_sort(test_dataset, gt_features)
        for n, t in enumerate(test_res_sorted):
            # We need to extract a random 1024-token chunk from the test track
            dec = process_ground_truth_track(t)
            # Grab the metadata JSON file
            meta = t.replace("piano_midi.mid", "metadata_tivo.json")
            meta_read = utils.read_json_cached(meta)
            # Get the first part of the ID
            mbz_id = meta_read["mbz_id"].split("-")[0]
            fp = os.path.join(EXAMPLES_DIR, f"{tok_fmt}_{str(n).zfill(3)}_real_{mbz_id}.mid")
            dec.dump_midi(fp)


if __name__ == "__main__":
    import argparse

    # Seed everything for reproducible results
    utils.seed_everything(utils.SEED)
    # Parsing arguments from the command line interface
    parser = argparse.ArgumentParser(description="Get N generated + real tracks for use in subjective listening test")
    parser.add_argument(
        "-c", "--generation-dir-clamp", type=str, help="Path to generated CLaMP MIDIs",
        default="reinforcement-customtok-plateau/"
                "reinforcement_iter3_customtok_10msmin_lineartime_moreaugment_"
                "init6e5reduce10patience5_batch4_1024seq_12l8h768d3072ff"
    )
    parser.add_argument(
        "-n", "--generation-dir-noclamp", type=str, help="Path to generated non-CLaMP MIDIs",
        default="finetuning-customtok-plateau/"
                "finetuning_customtok_10msmin_lineartime_moreaugment_"
                "init6e5reduce10patience5_batch4_1024seq_12l8h768d3072ff"
    )

    args = vars(parser.parse_args())
    main(generation_dir_clamp=args["generation_dir_clamp"], generation_dir_noclamp=args["generation_dir_noclamp"])
