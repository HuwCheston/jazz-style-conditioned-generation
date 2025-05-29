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

    def __init__(
            self,
            match_token: bool,
            files_paths: list[str],
            condition_token: str,
            clamp: clamp_utils.CLaMP3Model,
            n_tracks: int = None
    ):
        self.match_token = match_token
        super().__init__(files_paths, condition_token, clamp)
        if n_tracks is not None:
            self.files_paths = self.files_paths[:n_tracks]

    def get_tracks_with_condition(self, files_paths: list[str]):
        for track in files_paths:
            condition_tokens = self.get_condition_tokens(track)
            # Matching token: return all tracks with this token
            if self.match_token:
                if self.condition_token in condition_tokens:
                    yield track
            # Not matching token: return all valid tracks WITHOUT this token
            else:
                if (len([ct for ct in condition_tokens if ct in CONDITION_TOKENS]) > 0
                        and self.condition_token not in condition_tokens):
                    assert self.condition_token not in condition_tokens
                    yield track

    def __getitem__(self, item: int):
        # Return both the filepath and the extracted features
        fpath = self.files_paths[item]
        features = super().__getitem__(item)
        return fpath, features


class GenerationDataset(TestDataset):
    def __init__(
            self,
            match_token: bool,
            files_paths: list[str],
            condition_token: str,
            clamp: clamp_utils.CLaMP3Model,
            n_tracks: int = None
    ):
        super().__init__(match_token, files_paths, condition_token, clamp, n_tracks)

    @staticmethod
    def format_condition_token(condition_token) -> str:
        return utils.remove_punctuation(condition_token).replace(' ', '').lower()

    def get_tracks_with_condition(self, files_paths: list[str]):
        desired_tok_fmt = self.format_condition_token(self.condition_token)
        other_tok_fmt = tuple(self.format_condition_token(ct) for ct in CONDITION_TOKENS)

        for track in files_paths:
            # Skip over non-MIDI files
            if not track.endswith(".mid"):
                continue
            # Matching the condition token: return all tracks that start with the correct token
            track_gen_token = track.split(os.path.sep)[-1]
            if self.match_token:
                if track_gen_token.startswith(desired_tok_fmt):
                    yield track
            # NOT matching the condition token: return all other valid tracks
            else:
                if not track_gen_token.startswith(desired_tok_fmt):
                    if track_gen_token.startswith(other_tok_fmt):
                        yield track


def dump_real_track(input_filepath: str, output_filepath: str) -> str:
    """Given the filepath of a track, grab a random chunk of 1024 tokens and dump to disk with given filepath"""
    # Load and preprocess score in seconds
    loaded = preprocess_score(load_score(input_filepath, as_seconds=True))
    # Encode as tokens
    encoded = TOKENIZER(loaded)[0].ids
    # Get a random starting point for the 1024-token chunk
    #  Should be "within bounds" for the track
    start_point = np.random.randint(0, max(len(encoded) - MAX_SEQ_LEN, 0))
    end_point = start_point + MAX_SEQ_LEN
    chunk = encoded[start_point:end_point]
    # Back to a Score object to let us dump it to disk
    outp = TOKENIZER(torch.tensor([chunk]))
    # Grab the metadata JSON file
    meta = input_filepath.replace("piano_midi.mid", "metadata_tivo.json")
    meta_read = utils.read_json_cached(meta)
    # Get the first part of the ID
    mbz_id = meta_read["mbz_id"].split("-")[0]
    # Dump to disk with the complete filepath
    outpath = os.path.join(EXAMPLES_DIR, output_filepath + "_" + mbz_id + ".mid")
    outp.dump_midi(outpath)
    return outpath


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


def get_most_similar_track_idx(x_feat: torch.Tensor, y_feats: torch.Tensor, idx: int = 0) -> int:
    """Given a 1D tensor and 2D tensor of features, get index of row in 2D tensor that is most similar to 1D tensor"""
    match_sims = F.cosine_similarity(x_feat, y_feats, dim=-1)
    return torch.argsort(match_sims, descending=True).tolist()[idx]


def get_clamp_features_from_clip(clip_fpath: str, clamp) -> torch.Tensor:
    track_tmp = load_score(clip_fpath, as_seconds=True)
    track_loaded = preprocess_score(track_tmp)
    # Convert the ground truth track into the format required for CLaMP
    gt_data = clamp_utils.midi_to_clamp(track_loaded)
    # Extract features using CLaMP
    return clamp_utils.extract_clamp_features(gt_data, clamp)


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
            match_token=True,
            files_paths=HELD_OUT_TRACKS,
            condition_token=condition_token,
            clamp=clamp
        )

        # For every track in the test dataset, compute the cosine similarity vs the ground truth: shape (N_test_tracks)
        sim_matrix = torch.matmul(F.normalize(test_features, dim=1), F.normalize(gt_features, dim=1).T)
        collapsed = sim_matrix.mean(dim=1)
        # Extract top-N test tracks
        sort_idxs = collapsed.argsort(descending=True).tolist()[:N_TRACKS]
        test_fpaths_top_n = [test_fpaths[i] for i in sort_idxs]
        test_features_top_n = test_features[sort_idxs, :]

        # Get held-out test tracks WITHOUT this condition token
        test_nomatch_fpaths, test_nomatch_features = get_fpaths_and_features(
            TestDataset,
            match_token=False,
            files_paths=HELD_OUT_TRACKS,
            condition_token=condition_token,
            clamp=clamp
        )

        # Get generated tracks WITH this condition token, WITH clamp
        gen_match_clamp_fpaths, gen_match_clamp_features = get_fpaths_and_features(
            GenerationDataset,
            match_token=True,
            files_paths=gen_fps_clamp,
            condition_token=condition_token,
            clamp=clamp
        )
        # Get generated tracks WITHOUT this condition token, WITH clamp
        gen_nomatch_clamp_fpaths, gen_nomatch_clamp_features = get_fpaths_and_features(
            GenerationDataset,
            match_token=False,
            files_paths=gen_fps_clamp,
            condition_token=condition_token,
            clamp=clamp
        )
        # Get generated tracks WITH this condition token, WITHOUT clamp
        gen_match_noclamp_fpaths, gen_match_noclamp_features = get_fpaths_and_features(
            GenerationDataset,
            match_token=True,
            files_paths=gen_fps_noclamp,
            condition_token=condition_token,
            clamp=clamp
        )
        # Get generated tracks WITHOUT this condition token, WITHOUT clamp
        gen_nomatch_noclamp_fpaths, gen_nomatch_noclamp_features = get_fpaths_and_features(
            GenerationDataset,
            match_token=False,
            files_paths=gen_fps_noclamp,
            condition_token=condition_token,
            clamp=clamp
        )

        # Iterate over top-N held-out tracks that are most similar to the ground truth
        for n, (test_fpath, test_feat) in enumerate(zip(test_fpaths_top_n, test_features_top_n)):
            # Grab a random chunk of 1024 tokens and dump to disk
            test_fpath_out = dump_real_track(test_fpath, f'{tok_fmt}_{str(n).zfill(3)}_anchor')
            # Get features from the random chunk of 1024 tokens
            clip_features = get_clamp_features_from_clip(test_fpath_out, clamp)

            # Get all file paths for this track
            # Starting with the "real" tracks
            real_match = test_fpaths[get_most_similar_track_idx(clip_features, test_features, idx=0)]
            if real_match == test_fpath:
                real_match = test_fpaths[get_most_similar_track_idx(clip_features, test_features, idx=1)]
            real_nomatch = np.random.choice(test_nomatch_fpaths)

            # Then with generations using clamp
            gen_match_clamp = gen_match_clamp_fpaths[
                get_most_similar_track_idx(clip_features, gen_match_clamp_features)]
            gen_nomatch_clamp = np.random.choice(gen_nomatch_clamp_fpaths)

            # Then generations not using clamp
            gen_match_noclamp = gen_match_noclamp_fpaths[
                get_most_similar_track_idx(clip_features, gen_match_noclamp_features)]
            gen_nomatch_noclamp = np.random.choice(gen_nomatch_noclamp_fpaths)

            # Check all filepaths are unique
            all_fps = [
                real_match, real_nomatch,  # real tracks
                gen_match_clamp, gen_nomatch_clamp,  # generated tracks with clamp
                gen_match_noclamp, gen_nomatch_noclamp,  # generated tracks without clamp
                test_fpath
            ]
            assert len(set(all_fps)) == len(all_fps)

            # For the "real" tracks, we need to snip a random chunk of 1024 tokens. This function does that.
            dump_real_track(real_match, f'{tok_fmt}_{str(n).zfill(3)}_real_match')
            dump_real_track(real_nomatch, f'{tok_fmt}_{str(n).zfill(3)}_real_nomatch')

            # For the "generated" tracks, we can just use the full generation
            shutil.copy(gen_match_clamp, os.path.join(EXAMPLES_DIR, f'{tok_fmt}_{str(n).zfill(3)}_gen_match_clamp.mid'))
            shutil.copy(gen_match_noclamp,
                        os.path.join(EXAMPLES_DIR, f'{tok_fmt}_{str(n).zfill(3)}_gen_match_noclamp.mid'))
            shutil.copy(gen_nomatch_clamp,
                        os.path.join(EXAMPLES_DIR, f'{tok_fmt}_{str(n).zfill(3)}_gen_nomatch_clamp.mid'))
            shutil.copy(gen_nomatch_noclamp,
                        os.path.join(EXAMPLES_DIR, f'{tok_fmt}_{str(n).zfill(3)}_gen_nomatch_noclamp.mid'))


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
