#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for data augmentation"""

import os
import unittest

from symusic import Note

from jazz_style_conditioned_generation import utils
from jazz_style_conditioned_generation.data.augmentation import (
    get_pitch_augmentation_value,
    PITCH_AUGMENT_RANGE,
    data_augmentation,
    _data_augmentation_deterministic
)
from jazz_style_conditioned_generation.data.scores import load_score, note_list_to_score

TEST_RESOURCES = os.path.join(utils.get_project_root(), "tests/test_resources")
TEST_MIDI = os.path.join(TEST_RESOURCES, "test_midi1/piano_midi.mid")


class AugmentationTest(unittest.TestCase):
    def test_get_pitch_range(self):
        score = load_score(TEST_MIDI)
        expected_min_pitch, expected_max_pitch = 34, 98
        actual_min_pitch, actual_max_pitch = utils.get_pitch_range(score)
        self.assertEqual(expected_min_pitch, actual_min_pitch)
        self.assertEqual(expected_max_pitch, actual_max_pitch)

    def test_get_pitch_augment_value(self):
        # Test with a real midi clip
        score = load_score(TEST_MIDI)
        expected_max_augment = 3
        actual_augment = get_pitch_augmentation_value(score, PITCH_AUGMENT_RANGE)
        self.assertTrue(abs(actual_augment) <= expected_max_augment)
        # TODO: test with a midi clip that exceeds boundaries

    def test_data_augmentation(self):
        score = load_score(TEST_MIDI)
        prev_min_pitch, prev_max_pitch = 34, 98
        # Test with transposition up one semitone
        augmented = data_augmentation(
            score, pitch_augmentation_range=[1], duration_augmentation_range=[1.]
        )
        new_min_pitch, new_max_pitch = utils.get_pitch_range(augmented)
        self.assertEqual(new_min_pitch, prev_min_pitch + 1)
        self.assertEqual(new_max_pitch, prev_max_pitch + 1)
        # Test with transposition down two semitones
        augmented = data_augmentation(
            score, pitch_augmentation_range=[-2], duration_augmentation_range=[1.]
        )
        new_min_pitch, new_max_pitch = utils.get_pitch_range(augmented)
        self.assertEqual(new_min_pitch, prev_min_pitch - 2)
        self.assertEqual(new_max_pitch, prev_max_pitch - 2)
        # Test with shortening duration
        augmented = data_augmentation(
            score, pitch_augmentation_range=[0], duration_augmentation_range=[0.9]
        )
        self.assertLess(augmented.end(), score.end())
        # Test with increasing duration
        augmented = data_augmentation(
            score, pitch_augmentation_range=[0], duration_augmentation_range=[1.1]
        )
        self.assertGreater(augmented.end(), score.end())

    def test_deterministic_augmentation(self):
        # Test pitch augmentation: up five semitones
        notes = [
            Note(pitch=50, duration=10, time=1000, velocity=50, ttype="tick")
        ]
        sc = note_list_to_score(notes, 100)
        augment = _data_augmentation_deterministic(sc, 5, 1.0)
        expected_pitch = 55
        actual_pitch = augment.tracks[0].notes[0].pitch
        self.assertEqual(expected_pitch, actual_pitch)
        # Test pitch augmentation: down three semitones
        notes = [
            Note(pitch=50, duration=10, time=1000, velocity=50, ttype="tick"),
            Note(pitch=40, duration=10, time=1000, velocity=50, ttype="tick")
        ]
        sc = note_list_to_score(notes, 100)
        augment = _data_augmentation_deterministic(sc, -3, 0.9)
        expected_pitch = [47, 37]
        actual_pitch = [n.pitch for n in augment.tracks[0].notes]
        self.assertEqual(expected_pitch, actual_pitch)
        # Test duration augmentation
        notes = [
            Note(pitch=50, duration=100, time=1000, velocity=50, ttype="tick")
        ]
        sc = note_list_to_score(notes, 100)
        augment = _data_augmentation_deterministic(sc, 0, 0.5)
        expected_duration = 50
        actual_duration = augment.tracks[0].notes[0].duration
        self.assertEqual(expected_duration, actual_duration)
        expected_start = 500
        actual_start = augment.tracks[0].notes[0].time
        self.assertEqual(expected_start, actual_start)


if __name__ == '__main__':
    unittest.main()
