#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for utility functions"""

import os
import unittest

from symusic import Note

from jazz_style_conditioned_generation import utils
from jazz_style_conditioned_generation.data.dataloader import note_list_to_score


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
            with self.subTest(value=value):
                self.assertTrue(utils.string_to_bool(value))
        # Test for various falsy values
        falsy_values = ['no', 'false', 'f', 'n', '0']
        for value in falsy_values:
            with self.subTest(value=value):
                self.assertFalse(utils.string_to_bool(value))
        # Test for already boolean inputs
        self.assertTrue(utils.string_to_bool(True))
        self.assertFalse(utils.string_to_bool(False))
        # Test for invalid values that should raise ValueError
        invalid_values = ['maybe', 'something', '123', 'null', '', ]
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    utils.string_to_bool(value)


if __name__ == '__main__':
    unittest.main()
