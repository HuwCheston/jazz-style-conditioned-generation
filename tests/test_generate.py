#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for generate.py"""

import os
import unittest

from jazz_style_conditioned_generation import utils, generate
from jazz_style_conditioned_generation.data.tokenizer import (
    load_tokenizer,
    add_genres_to_vocab,
    add_tempos_to_vocab,
    add_pianists_to_vocab,
    add_timesignatures_to_vocab,
    add_recording_years_to_vocab
)


class TestGenerateDataset(unittest.TestCase):
    def setUp(self):
        self.TOKENIZER = load_tokenizer(tokenizer_str="tsd")
        add_tempos_to_vocab(self.TOKENIZER, min_tempo=80, )
        add_pianists_to_vocab(self.TOKENIZER)
        add_timesignatures_to_vocab(self.TOKENIZER, [3, 4])
        add_genres_to_vocab(self.TOKENIZER)
        add_recording_years_to_vocab(self.TOKENIZER)
        self.MIDI_PATH1 = os.path.join(utils.get_project_root(), "tests/test_resources/test_midi1/piano_midi.mid")

    def test_get_custom_conditioning_tokens(self):
        # Test with some custom pianist and genre tokens
        ds = generate.GenerateDataset(
            self.TOKENIZER,
            files_paths=[self.MIDI_PATH1],
            max_seq_len=utils.MAX_SEQUENCE_LENGTH,
            do_augmentation=False,
            do_conditioning=True,
            custom_pianists=["Bill Evans", "Herbie Hancock"],
            custom_genres=["Fusion", "Blues"],
            use_track_tokens=True,  # we'll use the track's tempo, year, + time signature
        )
        cts = ds.get_conditioning_tokens(utils.read_json_cached(ds.metadata_paths[0]))
        expected_tokens = [
            "GENRES_Fusion", "GENRES_Blues", "PIANIST_BillEvans", "PIANIST_HerbieHancock",
            "RECORDINGYEAR_1990", "TEMPOCUSTOM_299", "TIMESIGNATURECUSTOM_44"
        ]
        self.assertEqual(expected_tokens, [self.TOKENIZER[i] for i in cts])

        # Now, try again but don't use the track tokens
        ds.use_track_tokens = False
        cts = ds.get_conditioning_tokens(utils.read_json_cached(ds.metadata_paths[0]))
        expected_tokens = ["GENRES_Fusion", "GENRES_Blues", "PIANIST_BillEvans", "PIANIST_HerbieHancock", ]
        self.assertEqual(expected_tokens, [self.TOKENIZER[i] for i in cts])


if __name__ == '__main__':
    utils.seed_everything(utils.SEED)
    unittest.main()
