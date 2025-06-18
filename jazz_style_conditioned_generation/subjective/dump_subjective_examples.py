#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Dumps MIDI examples for subjective listening test"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from symusic import Score
from torch.utils.data import DataLoader
from tqdm import tqdm

from jazz_style_conditioned_generation import utils
from jazz_style_conditioned_generation.data.scores import (
    load_score,
    preprocess_score,
    get_notes_from_score,
    note_list_to_score
)
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
    "Bill Evans",
    "Oscar Peterson",
    "Keith Jarrett",
]
TRAIN_TRACKS = sorted(read_tracks_for_splits("train"))
TEST_TRACKS = read_tracks_for_splits("test")
VALID_TRACKS = read_tracks_for_splits("validation")
HELD_OUT_TRACKS = sorted(TEST_TRACKS + VALID_TRACKS)

N_TRACKS = 10
MAX_SEQ_LEN = 1024
MAX_DURATION_SECONDS = 15  # clip examples to 1024 tokens or 15 seconds, whichever comes first

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

    def __init__(
            self,
            files_paths: list[str],
            condition_token: str,
            clamp: clamp_utils.CLaMP3Model,
            n_tracks: int = None
    ):
        super().__init__(files_paths, condition_token, clamp)
        if n_tracks is not None:
            self.files_paths = self.files_paths[:n_tracks]

    def get_tracks_with_condition(self, files_paths: list[str]):
        for track in files_paths:
            condition_tokens = self.get_condition_tokens(track)
            # Matching the token we want
            if self.condition_token in condition_tokens:
                # Must not match with any of the other tokens we're testing
                # I.e., if the track is Avant Garde (which we want) but also Straight-Ahead, we must ignore it
                if not any([ct in CONDITION_TOKENS for ct in condition_tokens if ct != self.condition_token]):
                    yield track

    def __getitem__(self, item: int):
        # Return both the filepath and the extracted features
        fpath = self.files_paths[item]
        features = super().__getitem__(item)
        return fpath, features


class GenerationDataset(TestDataset):
    def __init__(
            self,
            files_paths: list[str],
            condition_token: str,
            clamp: clamp_utils.CLaMP3Model,
            n_tracks: int = None
    ):
        super().__init__(files_paths, condition_token, clamp, n_tracks)

    @staticmethod
    def format_condition_token(condition_token) -> str:
        return utils.remove_punctuation(condition_token).replace(' ', '').lower()

    def get_tracks_with_condition(self, files_paths: list[str]):
        desired_tok_fmt = self.format_condition_token(self.condition_token)
        for track in files_paths:
            # Skip over non-MIDI files
            if not track.endswith(".mid"):
                continue
            # Matching the condition token: return all tracks that start with the correct token
            track_gen_token = track.split(os.path.sep)[-1]
            if track_gen_token.startswith(desired_tok_fmt):
                # We don't need to check other condition tokens, each generation was only made with one condition token
                yield track


def process_midi(fpath: str) -> Score:
    """Loads, preprocesses, and truncates a MIDI file (generated or real). Returns a symusic.Score object."""
    # Load and preprocess score in seconds
    loaded = preprocess_score(load_score(fpath, as_seconds=True))
    # Encode as tokens
    encoded = TOKENIZER(loaded)[0].ids
    # If the chunk is too long in terms of tokens, get a random starting point
    if len(encoded) > MAX_SEQ_LEN:
        start_point = np.random.randint(0, max(len(encoded) - MAX_SEQ_LEN, 0))
        encoded = encoded[start_point:start_point + MAX_SEQ_LEN]
    assert len(encoded) <= MAX_SEQ_LEN
    # Back to a Score object
    outp = TOKENIZER(torch.tensor([encoded]))
    # If the chunk is too long in terms of seconds, truncate it to 15 seconds
    if outp.end() > MAX_DURATION_SECONDS:
        # Score -> Notes -> Truncate -> Score
        notes = get_notes_from_score(outp)
        notes_trunc = [n for n in notes if n.end <= MAX_DURATION_SECONDS]
        outp = note_list_to_score(notes_trunc, utils.TICKS_PER_QUARTER, ttype="Second")
        assert len(outp.tracks[0].notes) == len(notes_trunc)
    return outp


def process_metadata(fpath: str, condition_token: str, condition_type: str) -> dict:
    """Process the metadata for a track or return a dummy dictionary if we don't have this"""
    # Try and get the path to the metadata
    meta_read = dict(condition_token=condition_token, condition_type=condition_type)
    try:
        meta = fpath.replace("piano_midi.mid", "metadata_tivo.json")
        meta_read = utils.read_json_cached(meta)
    # Skip over tracks without metadata
    except (FileNotFoundError, UnicodeDecodeError):
        pass
    else:
        # Dump the metadata to disk if we have it
        meta_read["condition_token"] = condition_token
        meta_read["condition_type"] = condition_type
    return meta_read


def get_fpaths_and_features(dataset_cls, **dataset_kwargs) -> tuple[list[str], torch.Tensor]:
    """Given a dataset class and arguments, get all filepaths and features"""
    # Create the dataset with desired kwargs
    dataset = DataLoader(
        dataset_cls(**dataset_kwargs),
        shuffle=False,
        drop_last=False,
        batch_size=4
    )
    # Extract the features from all tracks in the dataset
    fpaths, features = zip(*[(fp, feat) for (fp, feat) in tqdm(dataset, desc="Extracting features...")])
    fpaths = [x for xs in fpaths for x in xs]  # flatten to (N_tracks)
    features = torch.cat(features, dim=0)  # shape (N_tracks, features)
    return fpaths, features


def get_most_similar_track_idxs(x_feats: torch.Tensor, y_feats: torch.Tensor) -> tuple:
    """Given 2D tensors features, get index of rows that are most similar"""
    # For every track in the test dataset, compute the cosine similarity vs the ground truth: shape (N_test_tracks)
    sim_matrix = torch.matmul(F.normalize(y_feats, dim=1), F.normalize(x_feats, dim=1).T)
    collapsed = sim_matrix.mean(dim=1)
    # Extract top-N test tracks
    idxs = collapsed.argsort(descending=True).tolist()[:N_TRACKS]
    sims = collapsed[idxs].tolist()
    return idxs, sims


def main(generation_dir_clamp: str, generation_dir_noclamp: str) -> None:
    clamp = clamp_utils.initialize_clamp(pretrained=True)
    generation_dir_clamp = os.path.join(utils.get_project_root(), "data/rl_generations", generation_dir_clamp)
    generation_dir_noclamp = os.path.join(utils.get_project_root(), "data/rl_generations", generation_dir_noclamp)

    # Load up generations made USING CLAMP
    gen_fps_clamp = sorted([
        os.path.join(generation_dir_clamp, i) for i in os.listdir(generation_dir_clamp)
        if i.endswith(".mid")  # should be a MIDI file
           and "iter003" in i  # should be generated with k=3 iterations of CLaMP-DPO loss
    ])
    utils.validate_paths(gen_fps_clamp, expected_extension=".mid")
    # Load up generations made WITHOUT CLAMP
    gen_fps_noclamp = sorted([
        os.path.join(generation_dir_noclamp, i) for i in os.listdir(generation_dir_noclamp)
        if i.endswith(".mid")  # should be a MIDI file
           and "iter000" in i  # should be generated with k=0 iterations of CLaMP-DPO loss
    ])
    utils.validate_paths(gen_fps_noclamp, expected_extension=".mid")

    for condition_token in CONDITION_TOKENS:
        tok_fmt = utils.remove_punctuation(condition_token).replace(' ', '').lower()
        logger.info(f"Processing token {tok_fmt}")
        # Get ground truth tracks from the training dataset with this condition
        gt_dataset = DataLoader(
            GroundTruthDataset(
                files_paths=TRAIN_TRACKS,
                condition_token=condition_token,
                clamp=clamp
            ),
            shuffle=False,
            drop_last=False,
            batch_size=4
        )
        # Extract the features
        gt_features = torch.cat([i for i in tqdm(gt_dataset, desc="Loading train examples...")], dim=0)

        # Get held-out test tracks with this condition token
        test_fpaths, test_features = get_fpaths_and_features(
            TestDataset,
            files_paths=HELD_OUT_TRACKS,
            condition_token=condition_token,
            clamp=clamp
        )
        # Get the idxs of the most similar tracks and index everything
        test_sort_idxs, test_sims = get_most_similar_track_idxs(gt_features, test_features)
        test_fpaths_top_n = [test_fpaths[i] for i in test_sort_idxs]

        # Get generated tracks WITH this condition token, WITH clamp
        gen_clamp_fpaths, gen_clamp_features = get_fpaths_and_features(
            GenerationDataset,
            files_paths=gen_fps_clamp,
            condition_token=condition_token,
            clamp=clamp
        )
        # Get the idxs of the most similar tracks and index everything
        gen_clamp_sort_idxs, gen_clamp_sims = get_most_similar_track_idxs(gt_features, gen_clamp_features)
        gen_clamp_fpaths_top_n = [gen_clamp_fpaths[i] for i in gen_clamp_sort_idxs]

        # Get generated tracks WITH this condition token, WITHOUT clamp
        gen_noclamp_fpaths, gen_noclamp_features = get_fpaths_and_features(
            GenerationDataset,
            files_paths=gen_fps_noclamp,
            condition_token=condition_token,
            clamp=clamp
        )
        # Get the idxs of the most similar tracks and index everything
        gen_noclamp_sort_idxs, gen_noclamp_sims = get_most_similar_track_idxs(gt_features, gen_noclamp_features)
        gen_noclamp_fpaths_top_n = [gen_noclamp_fpaths[i] for i in gen_noclamp_sort_idxs]

        # Iterate over all conditions: real, clamp, no_clamp
        all_fpaths = [test_fpaths_top_n, gen_clamp_fpaths_top_n, gen_noclamp_fpaths_top_n]
        all_sims = [test_sims, gen_clamp_sims, gen_noclamp_sims]
        for condition_type, fpaths, sims in zip(["real", "clamp", "noclamp"], all_fpaths, all_sims):
            # Iterate over all files for the condition with a counter
            for fnum, (fpath, sim) in enumerate(zip(fpaths, sims)):
                # Get the output path for saving this MIDI
                outpath_mid = os.path.join(EXAMPLES_DIR, f"{tok_fmt}_{str(fnum).zfill(3)}_{condition_type}.mid")
                outpath_js = outpath_mid.replace(".mid", ".json")
                # Write the metadata and dump the score
                if not os.path.isfile(outpath_js):
                    # Load up the metadata
                    meta_read = process_metadata(fpath, condition_token, condition_type)
                    meta_read["similarity"] = sim  # add in the cosine similarity value here
                    utils.write_json(meta_read, outpath_js)
                else:
                    logger.warning(f"File {outpath_js} exists, skipping!")
                if not os.path.isfile(outpath_mid):
                    # Convert the MIDI to a score and truncate to 1024 tokens/15 seconds duration
                    sc = process_midi(fpath)
                    sc.dump_midi(outpath_mid)
                else:
                    logger.warning(f"File {outpath_mid} exists, skipping!")


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
