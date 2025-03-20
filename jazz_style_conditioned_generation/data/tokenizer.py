#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Trains a Tokenizer in MIDITok and dumps a .json with all parameters"""

import os
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
from jazz_style_conditioned_generation.data.conditions import INCLUDE
from jazz_style_conditioned_generation.data.scores import load_score, preprocess_score

DEFAULT_TOKENIZER_CONFIG = {
    "pitch_range": (utils.MIDI_OFFSET, utils.MIDI_OFFSET + utils.PIANO_KEYS),
    "beat_res": {(0, utils.TIME_SIGNATURE): 100 // utils.TIME_SIGNATURE},  # 100 tokens per "bar", 10ms each
    "num_velocities": 32,
    "special_tokens": [
        "PAD",  # add for short inputs to ensure consistent sequence length for all inputs
        "BOS",  # beginning of sequence
        "EOS",  # end of sequence
        # "MASK",  # prevent attention to future tokens
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


class CustomTokTrainingIterator(TokTrainingIterator):
    """Modifies miditok.TokTrainingIterator to use our custom Score loading and preprocessing functions"""

    def __init__(
            self,
            tokenizer: MusicTokenizer,
            files_paths: Sequence[Path],
            tracks_idx_random_ratio_range: tuple[float, float] | None = None,
            bars_idx_random_ratio_range: tuple[float, float] | None = None,
    ):
        super().__init__(tokenizer, files_paths, tracks_idx_random_ratio_range, bars_idx_random_ratio_range)
        self.condition_tokens = [
            i for i in tokenizer.vocab if i.startswith(("GENRES", "PIANIST", "TEMPO", "TIMESIGNATURE"))
        ]

    def load_file(self, path: Path) -> list[str]:
        """Load a file and preprocess with our custom functions, then convert to a byte representation"""
        # Load the score using our custom loading function
        try:
            score = load_score(path)
        except SCORE_LOADING_EXCEPTION:
            return []
        # Apply our own preprocessing to the score
        score = preprocess_score(score)
        # Stuff below is copied from MIDITok.tokenizer_training_iterator.TokTrainingIterator unless indicated
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
        # REMOVED: stuff to do with MMM tokenization
        tokseq = self.tokenizer(
            score,
            encode_ids=False,
            no_preprocess_score=True,
            attribute_controls_indexes=ac_indexes,
        )
        # ADDED: check that no tokens are in the list of condition tokens
        for tok in tokseq[0].tokens:
            assert tok not in self.condition_tokens

        # REMOVED: splitting IDs (we don't want to do this ever)
        # Convert ids to bytes for training
        if isinstance(tokseq, TokSequence):
            token_ids = tokseq.ids
        else:
            token_ids = [seq.ids for seq in tokseq]
        bytes_ = self.tokenizer._ids_to_bytes(token_ids, as_one_str=True)
        if isinstance(bytes_, str):
            bytes_ = [bytes_]
        return bytes_


def add_pianists_to_vocab(tokenizer) -> None:
    """Adds all valid pianists on all tracks to the tokenizer"""
    for pianist in INCLUDE["pianist"]:
        # Add the tokenizer prefix here
        with_prefix = f'PIANIST_{utils.remove_punctuation(pianist).replace(" ", "")}'
        if with_prefix not in tokenizer.vocab:
            tokenizer.add_to_vocab(with_prefix, special_token=False)


def add_genres_to_vocab(tokenizer: MusicTokenizer) -> None:
    """Adds all valid genres for all tracks and artists to the tokenizer"""
    for genre in list(INCLUDE["genres"]):
        with_prefix = f'GENRES_{utils.remove_punctuation(genre).replace(" ", "")}'
        if with_prefix not in tokenizer.vocab:
            tokenizer.add_to_vocab(with_prefix, special_token=False)


def add_timesignatures_to_vocab(tokenizer: MusicTokenizer, time_signatures: list[int]) -> None:
    """Given a list of time signatures, add these to the vocabulary as custom tokens (shouldn't be used in decoding)"""
    for time_signature in time_signatures:
        tok_id = f'TIMESIGNATURECUSTOM_{time_signature}4'
        if tok_id not in tokenizer.vocab:
            tokenizer.add_to_vocab(tok_id, special_token=False)


def add_tempos_to_vocab(tokenizer: MusicTokenizer, min_tempo: int, n_tempos: int = 30, factor: float = 1.05) -> None:
    """Add tempo tokens to vocabulary using geometric distribution)"""
    # Create the geometric distribution
    tempo_range = [min_tempo]
    for _ in range(n_tempos - 1):
        tempo_range.append(round(tempo_range[-1] * factor))
    # Add the tokens
    for tempo in tempo_range:
        tok_id = f'TEMPOCUSTOM_{tempo}'
        if tok_id not in tokenizer.vocab:
            tokenizer.add_to_vocab(tok_id, special_token=False)


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


def load_tokenizer(**kwargs) -> MusicTokenizer:
    # Get the name of the tokenizer from the config dictionary
    tokenizer_method = kwargs.get("tokenizer_str", DEFAULT_TOKENIZER_CLASS)
    # Try and load a trained tokenizer
    tokenizer_path = kwargs.get("tokenizer_path", False)
    if os.path.isfile(tokenizer_path):
        logger.debug(f'Initialising tokenizer tupe {tokenizer_method} from path {tokenizer_path}')
        tokenizer = get_tokenizer_class_from_string(tokenizer_method)(params=tokenizer_path)
    # Otherwise, create the tokenizer from scratch
    else:
        tokenizer_kws = kwargs.get("tokenizer_kws", DEFAULT_TOKENIZER_CONFIG)
        # Add in any missing parameters with defaults
        tokenizer_kws = utils.update_dictionary(tokenizer_kws, DEFAULT_TOKENIZER_CONFIG)
        logger.debug(f'Initialising tokenizer type {tokenizer_method} with params {tokenizer_kws}')
        # Create the tokenizer configuration + tokenizer
        cfg = TokenizerConfig(**tokenizer_kws)
        tokenizer = get_tokenizer_class_from_string(tokenizer_method)(cfg)
    logger.debug(f'... got tokenizer: {tokenizer}')
    # We need to set this attribute to make decoding tokens easier
    #  At this point, the mapping just goes from token IDX -> [token IDX]
    #  However, if we train the tokenizer, we'll update it to go BPE token IDX -> [token1 IDX, token2 IDX]
    #  By setting it here, we ensure compatibility between trained + non-trained tokenizers when calculating
    #  evaluation metrics e.g. negative log-likelihood loss, accuracy scores etc.
    setattr(tokenizer, "bpe_token_mapping", {v: [v] for v in tokenizer.vocab.values()})
    return tokenizer


def train_tokenizer(tokenizer: MusicTokenizer, files_paths: list[str], **kwargs) -> None:
    """Trains a tokenizer given kwargs using our custom iterator class"""
    # We don't need to train a tokenizer if it's already been trained!
    if tokenizer.is_trained:
        logger.warning(f'... tried to train a tokenizer that has already been trained, skipping')
    else:
        # Get the parameters again from the dictionary
        training_method = kwargs.get("training_method", DEFAULT_TRAINING_METHOD)
        vocab_size = kwargs.get("vocab_size", DEFAULT_VOCAB_SIZE)

        # If we try to train with a smaller vocab size than the model currently has
        if vocab_size <= tokenizer.vocab_size:
            # Skip over training and just return without updating anything
            logger.warning(f'... tried to train a tokenizer with a smaller vocabulary size than it '
                           f'has already ({vocab_size} vs. {tokenizer.vocab_size}), skipping')
            return

        logger.debug(f'... training tokenizer with method {training_method}, vocab size {vocab_size}')
        # We need to train with our custom iterator so that we use our custom score loading + preprocessing functions
        utils.validate_paths(files_paths, expected_extension=".mid")
        tti = CustomTokTrainingIterator(tokenizer, files_paths)
        tokenizer.train(vocab_size=vocab_size, model=training_method, iterator=tti)
        logger.debug(f'... training finished: {tokenizer}')
    # Now, we update our token mapping to go BPE token1 IDX -> [token1 IDX, token2 IDX], ...
    #  This will trigger if the tokenizer has already been trained BEFORE calling this function, too
    bpe_token_mapping = {
        tokenizer.vocab_model[byt]: [tokenizer[t] for t in token_list]
        for byt, token_list in tqdm(tokenizer._vocab_learned_bytes_to_tokens.items(), desc="Creating token mapping...")
    }
    setattr(tokenizer, "bpe_token_mapping", bpe_token_mapping)


if __name__ == "__main__":
    tokfactory = load_tokenizer()
    midi_fps = utils.get_data_files_with_ext("data/raw", "**/*.mid")
    js_fps = [i.replace("piano_midi.mid", "metadata_tivo.json") for i in midi_fps]
    # Add genre tokens
    add_genres_to_vocab(tokfactory)
    gen_toks = [t for t in tokfactory.vocab.keys() if "GENRES" in t]
    print(f'Loaded {len(gen_toks)} genre tokens')
    print(sorted(set(gen_toks)))

    # Add pianist tokens
    add_pianists_to_vocab(tokfactory)
    pian_toks = [t for t in tokfactory.vocab.keys() if "PIANIST" in t]
    print(f'Loaded {len(pian_toks)} pianist tokens')
    print(pian_toks)

    # Train the tokenizer
    train_tokenizer(tokfactory, midi_fps)
