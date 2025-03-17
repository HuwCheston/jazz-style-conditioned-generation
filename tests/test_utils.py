#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for utility functions"""

import os
import unittest

from symusic import Note

from jazz_style_conditioned_generation import utils
from jazz_style_conditioned_generation.data.scores import note_list_to_score
from jazz_style_conditioned_generation.data.tokenizer import load_tokenizer, train_tokenizer


class TestUtils(unittest.TestCase):
    def test_update_dictionary(self):
        di1 = dict(t1=123, t2=345)
        di2 = dict(t1=321, t3=678)
        # Without replacement
        di1_up = utils.update_dictionary(di1, di2)
        for key in di2.keys():
            self.assertIn(key, list(di1_up.keys()))
        # With replacement
        di1_replace = utils.update_dictionary(di1, di2, overwrite=True)
        for key, value in di2.items():
            self.assertEqual(di1_replace[key], value)

    def test_get_pitch_range(self):
        note_list = [
            Note(pitch=50, duration=100, time=100, velocity=50, ttype="tick"),
            Note(pitch=60, duration=100, time=200, velocity=50, ttype="tick"),
            Note(pitch=70, duration=100, time=300, velocity=50, ttype="tick"),
        ]
        score = note_list_to_score(note_list, 100)
        expected = (50, 70)
        actual = utils.get_pitch_range(score)
        self.assertEqual(expected, actual)

    def test_get_project_root(self):
        expected_files = [".gitignore", "README.md", "requirements.txt", "setup.py"]
        root = utils.get_project_root()
        actual_files = os.listdir(root)
        for expected_file in expected_files:
            self.assertIn(expected_file, actual_files)

    def test_get_files_with_extension(self):
        expected_files = [
            os.path.join(utils.get_project_root(), "tests/test_resources/test_midi1/piano_midi.mid"),
            os.path.join(utils.get_project_root(), "tests/test_resources/test_midi_repeatnotes.mid")
        ]
        actual_files = utils.get_data_files_with_ext(dir_from_root="tests/test_resources")
        for expected_file in expected_files:
            self.assertIn(expected_file, actual_files)

    def test_string_to_bool(self):
        # Test for various truthy values
        truthy_values = ['yes', 'true', 't', 'y', '1']
        for value in truthy_values:
            self.assertTrue(utils.string_to_bool(value))
        # Test for various falsy values
        falsy_values = ['no', 'false', 'f', 'n', '0']
        for value in falsy_values:
            self.assertFalse(utils.string_to_bool(value))
        # Test for already boolean inputs
        self.assertTrue(utils.string_to_bool(True))
        self.assertFalse(utils.string_to_bool(False))
        # Test for invalid values that should raise ValueError
        invalid_values = ['maybe', 'something', '123', 'null', '', ]
        for value in invalid_values:
            with self.assertRaises(ValueError):
                utils.string_to_bool(value)

    @unittest.skipIf(os.getenv("REMOTE") == "true", "Skipping test on GitHub Actions")
    def test_skip_on_github_actions(self):
        self.assertTrue(len(os.listdir(os.path.join(utils.get_project_root(), "data/raw/pijama"))) > 100)

    def test_pad_sequence(self):
        # Test right padding
        tokseq = [2, 2, 2, 3, 4, 5]
        expected = [2, 2, 2, 3, 4, 5, 0, 0, 0, 0]
        actual = utils.pad_sequence(tokseq, desired_len=10, pad_token_id=0)
        self.assertEqual(actual, expected)
        # Test left padding
        expected = [0, 0, 0, 0, 2, 2, 2, 3, 4, 5]
        actual = utils.pad_sequence(tokseq, desired_len=10, pad_token_id=0, right_pad=False)
        self.assertEqual(actual, expected)

    def test_decode_bpe_sequence(self):
        # Get a random file
        dummy_file = os.path.join(utils.get_project_root(), "tests/test_resources/test_midi1/piano_midi.mid")
        # Load and train the tokenizer with this file
        dummy_tokenizer = load_tokenizer()
        train_tokenizer(dummy_tokenizer, [dummy_file], vocab_size=1000)
        # Encode as a token sequence
        dummy_tokseq = dummy_tokenizer.encode(dummy_file)
        dummy_ids = dummy_tokseq[0].ids
        # These token IDs shouldn't be in our base vocabulary
        with self.assertRaises(KeyError):
            for i in dummy_ids:
                _ = dummy_tokenizer[i]
        # Decode the tokenized results
        decoded_ids = utils.decode_bpe_sequence(dummy_ids, dummy_tokenizer)
        # The decoded sequence should be longer than the non-decoded sequence
        #  (i.e., BPE is compressing the length of our initial sequence)
        self.assertGreater(decoded_ids.size(1), len(dummy_ids))
        # All the decoded IDs should be in our base vocabulary
        for i in decoded_ids.squeeze(0).tolist():
            try:
                _ = dummy_tokenizer[i]
            except KeyError:
                raise self.fail()
            # We also shouldn't have any padding tokens as we've only passed in a batch of one item
            self.assertTrue(i != dummy_tokenizer.pad_token_id)


if __name__ == '__main__':
    unittest.main()
