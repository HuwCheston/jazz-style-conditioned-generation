#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Trains a Tokenizer in MIDITok and dumps a .json with all parameters"""

import json
import os
from datetime import datetime
from random import choice

from loguru import logger
from miditok import REMI, MIDILike, TSD, Structured, TokenizerConfig, constants

from jazz_style_conditioned_generation import utils
from jazz_style_conditioned_generation.data.conditions import get_special_tokens_for_condition

CONFIG = {
    "pitch_range": (21, 109),
    # "beat_res": {(0, 4): 8, (4, 12): 4},
    "num_velocities": 32,
    "special_tokens": [
        "PAD",  # add for short inputs to ensure consistent sequence length for all inputs
        "BOS",  # beginning of sequence
        "EOS",  # end of sequence
        "MASK",  # prevent attention to future tokens
        # conditioning tokens
        *get_special_tokens_for_condition("pianist", "PERF_"),
        *get_special_tokens_for_condition(["artist_genres", "album_genres"], "GENRE_"),
        *get_special_tokens_for_condition(["artist_moods", "album_moods"], "MOOD_"),
        *get_special_tokens_for_condition("album_themes", "THEME_")
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
VOCAB_SIZE = 30000
TRAINING_METHOD = "BPE"
TOKENIZER_CLASS = "TSD"

OUTPUT_DIR = os.path.join(utils.get_project_root(), 'outputs/tokenizers')
OUTPUT_MIDI_DIR = os.path.join(utils.get_project_root(), 'outputs/midi/tokenized')


def get_tokenizer_class_from_string(tokenizer_str: str = TOKENIZER_CLASS):
    """Maps a string reference to a tokenizer class to the actual MIDITok class"""
    if tokenizer_str == "REMI":
        return REMI
    elif tokenizer_str == "MIDILike":
        return MIDILike
    elif tokenizer_str == "TSD":
        return TSD
    elif tokenizer_str == "Structured":
        return Structured
    else:
        raise ValueError(f'`tokenizer_str` {tokenizer_str} is not recognized')


def load_saved_tokenizer(tokenizer_str: str, training_method: str, tokenizer_config: dict):
    """Tries to load a saved tokenizer with the given parameters and config, raises FileNotFoundError when none found"""

    def check_inner():
        """Check inner key-value pairs for a tokenizer config"""
        for k, v in js_loaded['config'].items():
            try:
                try:
                    _ = iter(v)
                # If the value is NON-ITERABLE, we can compare directly
                except TypeError:
                    yield v == tokenizer_config[k]
                # MIDITok typically adds some extra information to iterable values, so we can't compare them
                else:
                    continue
            except KeyError:
                continue

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
                *check_inner()  # all non-iterable parameters should be the same
        )):
            # Load the tokenizer using these saved parameters
            logger.info(f'loaded tokenizer at {tokenizer_js}!')
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
    tokenizer = get_tokenizer_class_from_string(tokenizer_str)(TokenizerConfig(**tokenizer_config))
    # Train the tokenizer on our MIDI paths
    tokenizer.train(vocab_size=VOCAB_SIZE, model=training_method, files_paths=midi_paths)
    # Dump the tokenizer instance
    now = datetime.now().strftime('%y_%m_%d_%H:%M:%S')  # add the time the tokenizer was created to the filename
    tokenizer.save(
        os.path.join(OUTPUT_DIR, f'{tokenizer_str.lower()}_{VOCAB_SIZE}_{training_method.lower()}_{now}.json')
    )
    return tokenizer


def get_tokenizer(tokenizer_str: str, training_method: str, tokenizer_config: dict):
    """Given a tokenizer name (as string), training method, and config, load a saved tokenizer or train from scratch"""
    try:
        tokenizer = load_saved_tokenizer(tokenizer_str, training_method, tokenizer_config)
    except FileNotFoundError:
        with utils.timer('train tokenizer'):
            tokenizer = train_tokenizer(tokenizer_str, training_method, tokenizer_config)
    finally:
        logger.info(f'tokenizer created with parameters: {tokenizer.__repr__()}')
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


def main():
    # Load the tokenizer with default arguments
    tokenizer = get_tokenizer(TOKENIZER_CLASS, TRAINING_METHOD, CONFIG)
    # Encode and decode a few example MIDIs
    for i in range(5):
        encode_decode_midi(tokenizer)


if __name__ == "__main__":
    utils.seed_everything()
    main()
