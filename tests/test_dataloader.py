#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for dataloader"""

import unittest

import torch

from jazz_style_conditioned_generation import utils
from jazz_style_conditioned_generation.data.dataloader import DatasetMIDICondition


class DataloaderTest(unittest.TestCase):
    DS = DatasetMIDICondition(
        condition_mapping={
            "genres": {
                "Acoustic Jazz": 0,
                "African Folk": 1,
                "Math Rock": 2,
            },
            "pianist": {
                "Jimmy Jim": 0,
                "Bobby Bob": 1,
                "Freddie Fred": 2
            }
        },
        combine_artist_and_album_tags=False,
        files_paths=["a/fake/filepath"],
        tokenizer=None,
        max_seq_len=utils.MAX_SEQUENCE_LENGTH,
        bos_token_id=1,
        eos_token_id=2,
    )

    def test_get_conditions_from_metadata(self):
        dummy_metadata = {
            "album_genres": [
                {"name": "Math Rock"},
                {"name": "African Folk"},
            ],
            "artist_genres": [
                {"name": "Acoustic Jazz"}
            ],
            "album_themes": [
                {"name": "Themey Theme"}
            ],
            "pianist": "Freddie Fred"
        }
        # First, we just want to gather the album-level genres
        expected = {"genres": ["Math Rock", "African Folk"], "pianist": ["Freddie Fred"]}
        actual = self.DS.get_conditions(dummy_metadata)
        self.assertEqual(expected, actual)
        # Now, we want to gather both the artist-level genre tags and the album-level genre tags
        self.DS.combine_artist_and_album_tags = True
        expected = {"genres": ["Math Rock", "African Folk", "Acoustic Jazz"], "pianist": ["Freddie Fred"]}
        actual = self.DS.get_conditions(dummy_metadata)
        self.assertEqual(expected, actual)
        # Set back to default
        self.DS.combine_artist_and_album_tags = False

    def test_conditions_to_tokens(self):
        dummy_results = {"genres": ["Math Rock", "African Folk"], "pianist": ["Freddie Fred"]}
        expected = ["GENRES_2", "GENRES_1", "PIANIST_2"]
        actual = self.DS.conditions_to_tokens(dummy_results)
        self.assertEqual(expected, actual)

    def test_ensemble_token(self):
        # Test with JTD
        fname = "a/fake/filepath/jtd/track/piano_midi.mid"
        expected = "ENSEMBLE_0"
        actual = self.DS.get_ensemble_context_token(fname)
        self.assertEqual(expected, actual)
        # Test with PiJama
        fname = "a/fake/filepath/pijama/track/piano_midi.mid"
        expected = "ENSEMBLE_1"
        actual = self.DS.get_ensemble_context_token(fname)
        self.assertEqual(expected, actual)
        # Should raise an error
        bad_fname = "a/fake/filepath/with/no/dataset/name"
        self.assertRaises(ValueError, self.DS.get_ensemble_context_token, bad_fname)

    def test_add_special_tokens(self):
        input_ids = torch.tensor([1, 2, 3])
        special_tokens = torch.tensor([5, 5, 5])
        # By default, we add at position 1, which is the "beginning of sequence" tag
        expected = [1, 5, 5, 5, 2, 3]
        actual = utils.add_to_tensor_at_idx(input_ids, special_tokens).tolist()
        self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
