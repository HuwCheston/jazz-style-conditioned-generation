#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Trains a Tokenizer in MIDITok and dumps a .json with all parameters"""

import os
from itertools import product
from pathlib import Path
from typing import Sequence

import numpy as np
from loguru import logger
from miditok import MusicTokenizer, TokSequence, TokenizerConfig
from miditok.attribute_controls import create_random_ac_indexes
from miditok.constants import SCORE_LOADING_EXCEPTION
from miditok.tokenizations import REMI, MIDILike, TSD, Structured, PerTok
from miditok.tokenizer_training_iterator import TokTrainingIterator
from tqdm import tqdm

from jazz_style_conditioned_generation import utils
from jazz_style_conditioned_generation.data.augmentation import (
    _data_augmentation_deterministic,
    PITCH_AUGMENT_RANGE,
    DURATION_AUGMENT_RANGE
)
from jazz_style_conditioned_generation.data.scores import load_score, preprocess_score

DEFAULT_TOKENIZER_CONFIG = {
    "pitch_range": (utils.MIDI_OFFSET, utils.MIDI_OFFSET + utils.PIANO_KEYS),
    "beat_res": {(0, utils.TIME_SIGNATURE): 100 // utils.TIME_SIGNATURE},  # 100 tokens per "bar", 10ms each
    "num_velocities": 32,
    "special_tokens": [
        "PAD",  # add for short inputs to ensure consistent sequence length for all inputs
        "BOS",  # beginning of sequence
        "EOS",  # end of sequence
        "MASK",  # prevent attention to future tokens
    ],
    "use_chords": False,
    "use_rests": False,
    "use_tempos": False,
    "use_time_signatures": False,
    "use_programs": False,
    "use_sustain_pedals": False,
    "use_pitch_bends": False,
    "use_velocities": True,
    # "chord_maps": constants.CHORD_MAPS,  # TODO: think more about this
    "remove_duplicated_notes": True,
    "encode_ids_split": "no",
    "use_pitchdrum_tokens": False,
    "programs": [0],  # only piano
}
DEFAULT_VOCAB_SIZE = 1000
DEFAULT_TRAINING_METHOD = "BPE"
DEFAULT_TOKENIZER_CLASS = "tsd"

OUTPUT_MIDI_DIR = os.path.join(utils.get_project_root(), 'outputs/midi/tokenized')


class TokTrainingIteratorAugmentation(TokTrainingIterator):
    """Allows a tokenizer to be trained using AUGMENTED versions of MIDI files, generated on the fly"""

    def __init__(
            self,
            tokenizer: MusicTokenizer,
            files_paths: Sequence[Path],
            tracks_idx_random_ratio_range: tuple[float, float] | None = None,
            bars_idx_random_ratio_range: tuple[float, float] | None = None,
    ):
        super().__init__(tokenizer, files_paths, tracks_idx_random_ratio_range, bars_idx_random_ratio_range)
        self.files_paths_with_augs = list(self.get_augs_for_files())

    def get_augs_for_files(self):
        """Iterates through all files and returns tuples of (filepath, pitch_augmentation, duration_augmentation)"""
        for f in tqdm(self.files_paths, desc=f"Getting all augmentations for {len(self.files_paths)} tracks..."):
            # Load as a score object
            sc = load_score(f)
            # Apply our preprocessing to remove invalid notes, merge repeated notes etc.
            preproc = preprocess_score(sc)
            # Get the minimum and maximum pitch of the preprocessed scores
            min_pitch, max_pitch = utils.get_pitch_range(preproc)
            # Get the range of potential pitch augmentations that can be applied to this score
            pitch_augs = [
                i for i in PITCH_AUGMENT_RANGE
                if max_pitch + i <= utils.MIDI_OFFSET + utils.PIANO_KEYS
                   and min_pitch + i >= utils.MIDI_OFFSET
            ]
            # Iterate over all the possible augmentations we can apply to the score and yield a tuple
            for pitch_aug, duration_aug in product(pitch_augs, DURATION_AUGMENT_RANGE):
                yield f, pitch_aug, duration_aug

    def load_file_with_aug(self, path: Path, pitch_aug: int, dur_aug: float) -> list[str]:
        """Load a file, apply augmentation, and convert to a byte representation"""
        # Load and tokenize file
        try:
            score = load_score(path)
        except SCORE_LOADING_EXCEPTION:
            return []

        # Apply our own preprocessing to the score
        score = preprocess_score(score)
        # Apply the desired augmentations to the score
        score = _data_augmentation_deterministic(score, pitch_aug, dur_aug)

        # Everything below is copied from MIDITok.tokenizer_training_iterator.TokTrainingIterator
        # Preprocess first to already have the appropriate tracks idx in case of deletes
        score = self.tokenizer.preprocess_score(score)

        # Randomly create attribute controls indexes
        ac_indexes = None
        if (
                len(self.tracks_idx_random_ratio_range) > 0
                or len(self.bars_idx_random_ratio_range) > 0
        ):
            ac_indexes = create_random_ac_indexes(
                score,
                self.tokenizer.attribute_controls,
                self.tracks_idx_random_ratio_range,
                self.bars_idx_random_ratio_range,
            )

        # Tokenize the file
        # Need to specify `encode_ids=False` as it might be already pretrained
        # For MMM, we make sure to have sequences separated per track
        kwargs = {}
        # can't use isinstance because of circular import
        if type(self.tokenizer).__name__ == "MMM":
            kwargs["concatenate_track_sequences"] = False
        tokseq = self.tokenizer(
            score,
            encode_ids=False,
            no_preprocess_score=True,
            attribute_controls_indexes=ac_indexes,
            **kwargs,
        )

        # Split ids if requested
        if self.tokenizer.config.encode_ids_split in ["bar", "beat"]:
            if isinstance(tokseq, TokSequence):
                tokseq = [tokseq]

            new_seqs = []
            for seq in tokseq:
                if self.tokenizer.config.encode_ids_split == "bar":
                    new_seqs += seq.split_per_bars()
                else:
                    new_seqs += seq.split_per_beats()
            tokseq = [seq for seq in new_seqs if len(seq) > 0]

        # Convert ids to bytes for training
        if isinstance(tokseq, TokSequence):
            token_ids = tokseq.ids
        else:
            token_ids = [seq.ids for seq in tokseq]
        bytes_ = self.tokenizer._ids_to_bytes(token_ids, as_one_str=True)
        if isinstance(bytes_, str):
            bytes_ = [bytes_]

        return bytes_

    def __len__(self):
        return len(self.files_paths_with_augs)

    def __getitem__(self, idx: int) -> list[str]:
        return self.load_file_with_aug(*self.files_paths_with_augs[idx])

    def __str__(self) -> str:
        """Return the ``str`` representation of the iterator."""
        return f"{self.tokenizer} - {len(self.files_paths)} tracks, {len(self)} augmented tracks"


def add_conditions_to_vocab(tokenizer: MusicTokenizer, condition_mapping: dict) -> None:
    """Given a mapping with form {condition_type: {condition1: token1, ...}}, add tokens to tokenizer"""
    for mapping in condition_mapping.values():
        for token in mapping.values():
            tokenizer.add_to_vocab(token)
    # No need to return, add_to_vocab works inplace


def add_timesignatures_to_vocab(tokenizer: MusicTokenizer, time_signatures: list[int]) -> None:
    """Given a list of time signatures, add these to the vocabulary as custom tokens (shouldn't be used in decoding)"""
    for time_signature in time_signatures:
        tok_id = f'TIMESIGNATURECUSTOM_{time_signature}4'
        tokenizer.add_to_vocab(tok_id)


def add_tempos_to_vocab(tokenizer: MusicTokenizer, tempo_range: tuple, n_tempos: int = 32) -> None:
    """Given a range of tempos, add these to the vocabulary as custom tokens (shouldn't be used in decoding)"""
    tempo_range = np.linspace(*tempo_range, n_tempos).round().astype(int)
    for tempo in tempo_range:
        tok_id = f'TEMPOCUSTOM_{tempo}'
        tokenizer.add_to_vocab(tok_id)


def get_tokenizer_class_from_string(tokenizer_type: str):
    """Given a string, return the correct tokenizer class"""
    valids = ["remi", "midilike", "tsd", "structured", "pertok"]
    tokenizer_type = tokenizer_type.lower()
    if tokenizer_type == "remi":
        return REMI
    elif tokenizer_type == "midilike":
        return MIDILike
    elif tokenizer_type == "tsd":
        return TSD
    elif tokenizer_type == "structured":
        return Structured
    elif tokenizer_type == "pertok":
        return PerTok
    else:
        raise ValueError(f'`tokenizer_type` must be one of {", ".join(valids)} but got {tokenizer_type}')


def fix_pertok_microtiming_bug(tokenizer: PerTok) -> None:
    """Fixes https://github.com/Natooz/MidiTok/issues/227 by setting a missing attribute for the PerTok tokenizer"""
    if not hasattr(tokenizer, "microtiming_tick_values"):
        # This just copies the code from miditok.tokenizers.pertok, line 138
        mt_bins = tokenizer.config.additional_params["num_microtiming_bins"]
        tokenizer.microtiming_tick_values = np.linspace(
            -tokenizer.max_mt_shift,
            tokenizer.max_mt_shift,
            mt_bins + 1,
            dtype=np.intc
        )
        assert hasattr(tokenizer, "microtiming_tick_values")


def load_or_train_tokenizer(
        tokenizer_path: str,
        tokenizer_cfg: dict,
        track_paths: list[str]
) -> MusicTokenizer:
    """Tries to load a tokenizer from `tokenizer_path`, trains from scratch if this cannot be found"""

    # Get the variables from the config dictionary
    tokenizer_method = tokenizer_cfg.get("tokenizer_str", DEFAULT_TOKENIZER_CLASS)
    tokenizer_kws = tokenizer_cfg.get("tokenizer_kws", DEFAULT_TOKENIZER_CONFIG)
    # Add in any missing parameters with defaults
    tokenizer_kws = utils.update_dictionary(tokenizer_kws, DEFAULT_TOKENIZER_CONFIG)
    logger.debug(f'Initialising tokenizer type {tokenizer_method} with params {tokenizer_kws}')
    # If we've already trained the tokenizer for this run
    if os.path.isfile(tokenizer_path):
        tokenizer = get_tokenizer_class_from_string(tokenizer_method)(params=tokenizer_path)
        logger.debug(f'... loading tokenizer from {tokenizer_path}')
        # Fix a MIDITok bug related to loading a pre-trained PerTok tokenizer
        if tokenizer_method == "pertok":
            fix_pertok_microtiming_bug(tokenizer)

    # Otherwise, we need to create the tokenizer from scratch
    else:
        logger.debug('... could not find saved tokenizer for this experiment/run, creating it from scratch!')
        cfg = TokenizerConfig(**tokenizer_kws)
        tokenizer = get_tokenizer_class_from_string(tokenizer_method)(cfg)
        logger.debug(f'... got tokenizer: {tokenizer}')
        # If we want to train the tokenizer
        if tokenizer_cfg.get("do_training", False):
            # Get the parameters again from the dictionary
            training_method = tokenizer_cfg.get("training_method", DEFAULT_TRAINING_METHOD)
            vocab_size = tokenizer_cfg.get("vocab_size", DEFAULT_VOCAB_SIZE)
            logger.debug(f'... training tokenizer with method {training_method}, vocab size {vocab_size}')
            # TODO: this should use our custom load_score function
            # TODO: we need to add our condition tokens in BEFORE training the tokenizer or else we'll get errors
            tokenizer.train(vocab_size=vocab_size, model=training_method, files_paths=track_paths)

            # Create the iterator which we use to train with augmentation
            # tti = TokTrainingIteratorAugmentation(tokenizer, track_paths)
            # logger.debug(f'... using iterator: {tti}')
            # # Train the tokenizer
            # tokenizer.train(vocab_size=vocab_size, model=training_method, iterator=tti)
            logger.debug(f'... successfully trained tokenizer')
        # Finally, we can dump the tokenizer so that we reload it on future runs
        tokenizer.save(tokenizer_path)
        logger.debug(f'... tokenizer saved to {tokenizer_path}')
    return tokenizer


if __name__ == "__main__":
    tokfactory = REMI()
    midi_fps = utils.get_data_files_with_ext("data/raw", "**/*.mid")[:100]
    ts = TokTrainingIteratorAugmentation(tokfactory, midi_fps)
    print(ts)
