#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Trains a Tokenizer in MIDITok and dumps a .json with all parameters"""

import json
import os
from random import choice

from loguru import logger
from miditok import REMI, MIDILike, TSD, PerTok, Structured, TokenizerConfig, constants

from jazz_style_conditioned_generation import utils

DEFAULT_TOKENIZER_CONFIG = {
    "pitch_range": (21, 109),
    # "beat_res": {(0, 4): 8, (4, 12): 4},
    "num_velocities": 32,
    "special_tokens": [
        "PAD",  # add for short inputs to ensure consistent sequence length for all inputs
        "BOS",  # beginning of sequence
        "EOS",  # end of sequence
        "MASK",  # prevent attention to future tokens
    ],
    "use_chords": True,
    "use_rests": True,
    "use_tempos": False,
    "use_time_signatures": False,
    "use_programs": False,
    "use_sustain_pedals": False,
    "use_pitch_bends": False,
    "chord_maps": constants.CHORD_MAPS,  # TODO: think more about this
    "remove_duplicated_notes": True,
}
# TODO: add a way to parameterize these
VOCAB_SIZE = 20000
DEFAULT_TRAINING_METHOD = "BPE"
# TODO: we should almost definitely use the `PerTok` tokenizer here
DEFAULT_TOKENIZER_CLASS = "TSD"

OUTPUT_DIR = os.path.join(utils.get_project_root(), 'outputs/tokenizers')
OUTPUT_MIDI_DIR = os.path.join(utils.get_project_root(), 'outputs/midi/tokenized')


def get_tokenizer_class_from_string(tokenizer_str: str = DEFAULT_TOKENIZER_CLASS):
    """Maps a string reference to a tokenizer class to the actual MIDITok class"""
    if tokenizer_str == "REMI":
        return REMI
    elif tokenizer_str == "MIDILike":
        return MIDILike
    elif tokenizer_str == "TSD":
        return TSD
    elif tokenizer_str == "Structured":
        return Structured
    elif tokenizer_str == "PerTok":
        return PerTok
    else:
        raise ValueError(f'`tokenizer_str` {tokenizer_str} is not recognized')


def check_loaded_config(loaded_config: dict, desired_config: dict):
    """Check inner key-value pairs for a tokenizer config"""
    for k, v in loaded_config.items():
        try:
            try:
                _ = iter(v)
            # If the value is NON-ITERABLE, we can compare directly
            except TypeError:
                yield v == desired_config[k]
            # MIDITok typically adds some extra information to iterable values, so we can't compare them
            else:
                continue
        except KeyError:
            continue


def load_saved_tokenizer(tokenizer_str: str, training_method: str, tokenizer_config: dict):
    """Tries to load a saved tokenizer with the given parameters and config, raises FileNotFoundError when none found"""

    for tokenizer_js in os.listdir(OUTPUT_DIR):
        if not tokenizer_js.endswith('.json'):  # skip over any MIDI files, for example
            continue
        # Read the dictionary saved for this tokenizer
        js_loaded = utils.read_json_cached(os.path.join(OUTPUT_DIR, tokenizer_js))
        # If the saved tokenizer has all the parameters we want to use
        if all((
                js_loaded["tokenization"] == tokenizer_str,  # class should be the same
                json.loads(js_loaded["_model"])['model']['type'] == training_method,
                # training method should be the same
                *check_loaded_config(js_loaded["config"], tokenizer_config)
        # all non-iterable parameters should be the same
        )):
            # Load the tokenizer using these saved parameters
            logger.info(f'Loaded tokenizer at {tokenizer_js}!')
            return get_tokenizer_class_from_string(tokenizer_str)(params=os.path.join(OUTPUT_DIR, tokenizer_js))
        # Otherwise, try the next tokenizer
        else:
            continue
    raise FileNotFoundError(f'Could not find valid any tokenizer in {OUTPUT_DIR}')


def train_tokenizer(tokenizer_str: str, training_method: str, tokenizer_config: dict):
    """Train a tokenizer from scratch using all MIDI files in the dataset"""
    # Get filepaths for all MIDI files in the /data/raw/ directories
    midi_paths = utils.get_data_files_with_ext(ext="**/*.mid")
    # Instantiate the tokenizer instance with the passed parameters
    tc = TokenizerConfig(**tokenizer_config)
    tokenizer = get_tokenizer_class_from_string(tokenizer_str)(tc)
    # Train the tokenizer on our MIDI paths
    tokenizer.train(vocab_size=VOCAB_SIZE, model=training_method, files_paths=midi_paths)
    # Dump the tokenizer instance
    tokenizer.save(
        os.path.join(OUTPUT_DIR, f'{tokenizer_str.lower()}_{VOCAB_SIZE}_{training_method.lower()}_{utils.now()}.json')
    )
    return tokenizer


def get_tokenizer(
        tokenizer_str: str = DEFAULT_TOKENIZER_CLASS,
        training_method: str = DEFAULT_TRAINING_METHOD,
        tokenizer_config: dict = None
):
    """Given a tokenizer name (as string), training method, and config, load a saved tokenizer or train from scratch"""
    if tokenizer_config is None:
        tokenizer_config = DEFAULT_TOKENIZER_CONFIG

    try:
        tokenizer = load_saved_tokenizer(tokenizer_str, training_method, tokenizer_config)
    except FileNotFoundError:
        with utils.timer('train tokenizer'):
            tokenizer = train_tokenizer(tokenizer_str, training_method, tokenizer_config)
    finally:
        # TODO: errors can be a bit cryptic here
        logger.info(f'Tokenizer created with parameters: {tokenizer.__repr__()}')
        return tokenizer


def encode_decode_midi(tokenizer, midi_fpath: str = None):
    """Encodes a MIDI with a tokenizer, decodes it, then saves. For sanity checking the tokenization process."""
    # If we don't pass in a MIDI file, make a random selection from all the available MIDI files
    if midi_fpath is None:
        midi_paths = utils.get_data_files_with_ext(ext="**/*.mid")
        midi_fpath = choice(midi_paths)
    # Make sure that the path exists
    assert os.path.exists(midi_fpath), f'Could not find MIDI at {midi_fpath}'
    # Encode as tokens and then decode back to MIDI
    tokens = tokenizer(midi_fpath)
    decoded = tokenizer(tokens)
    # Dump the MIDI into the output directory
    out_fname = midi_fpath.split(os.path.sep)[-2] + '_encoded.mid'
    decoded.dump_midi(os.path.join(OUTPUT_MIDI_DIR, out_fname))


if __name__ == "__main__":
    utils.seed_everything()
    # Load the tokenizer with default arguments
    token_factory = get_tokenizer(DEFAULT_TOKENIZER_CLASS, DEFAULT_TRAINING_METHOD, DEFAULT_TOKENIZER_CONFIG)
    # Encode and decode a few example MIDIs
    for i in range(5):
        encode_decode_midi(token_factory)
