#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Trains a Tokenizer in MIDITok and dumps a .json with all parameters"""

import os
from random import choice

from jazz_style_conditioned_generation import utils

DEFAULT_TOKENIZER_CONFIG = {
    "pitch_range": (utils.MIDI_OFFSET, utils.MIDI_OFFSET + utils.PIANO_KEYS),
    "beat_res": {(0, 4): 8, (4, 12): 8},
    "num_velocities": 32,
    "special_tokens": [
        "PAD",  # add for short inputs to ensure consistent sequence length for all inputs
        "BOS",  # beginning of sequence
        "EOS",  # end of sequence
        "MASK",  # prevent attention to future tokens
    ],
    "use_chords": False,
    "use_rests": True,
    "use_tempos": False,
    "use_time_signatures": False,
    "use_programs": False,
    "use_sustain_pedals": False,
    "use_pitch_bends": False,
    # "chord_maps": constants.CHORD_MAPS,  # TODO: think more about this
    "remove_duplicated_notes": True,
}
DEFAULT_VOCAB_SIZE = 1000
DEFAULT_TRAINING_METHOD = "BPE"
DEFAULT_TOKENIZER_CLASS = "TSD"

OUTPUT_MIDI_DIR = os.path.join(utils.get_project_root(), 'outputs/midi/tokenized')


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
