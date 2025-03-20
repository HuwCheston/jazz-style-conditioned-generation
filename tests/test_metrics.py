#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for evaluation metrics"""

import os
import unittest

import numpy as np
import torch
from miditok import MIDILike, TokenizerConfig
from muspy import Music
from symusic import Note

from jazz_style_conditioned_generation import metrics
from jazz_style_conditioned_generation import utils
from jazz_style_conditioned_generation.data.scores import note_list_to_score
from jazz_style_conditioned_generation.data.tokenizer import load_tokenizer


class MetricsTest(unittest.TestCase):
    def test_to_muspy(self):
        notes = [
            Note(pitch=80, time=100, duration=100, velocity=80, ttype="tick"),
            Note(pitch=70, time=200, duration=100, velocity=80, ttype="tick"),
            Note(pitch=60, time=300, duration=100, velocity=80, ttype="tick"),
        ]
        score = note_list_to_score(notes, ticks_per_quarter=100)
        muspy = metrics._symusic_to_muspy(score)
        self.assertTrue(isinstance(muspy, Music))
        self.assertTrue(muspy.resolution == score.ticks_per_quarter)
        self.assertTrue(len(muspy.tracks[0].notes) == len(score.tracks[0].notes))
        self.assertTrue(muspy.get_end_time() == score.end())

    def test_coerce_to_muspy_wrapper(self):
        @metrics.coerce_to_muspy
        def func(x):
            return x

        fp = os.path.join(utils.get_project_root(), "tests/test_resources/test_midi1/piano_midi.mid")
        self.assertTrue(isinstance(func(fp), Music))
        notes = [
            Note(pitch=80, time=100, duration=100, velocity=80, ttype="tick"),
            Note(pitch=70, time=200, duration=100, velocity=80, ttype="tick"),
            Note(pitch=60, time=300, duration=100, velocity=80, ttype="tick"),
        ]
        score = note_list_to_score(notes, ticks_per_quarter=100)
        self.assertTrue(isinstance(func(score), Music))
        self.assertTrue(isinstance(func(Music()), Music))
        with self.assertRaises(TypeError):
            _ = func(123)
            _ = func(0.1)
            _ = func(True)

    def test_catch_zero_division(self):
        @metrics.catch_zero_division
        def good():
            return 1 / 2

        @metrics.catch_zero_division
        def bad():
            return 1 / 0

        self.assertTrue(good() == 0.5)
        self.assertTrue(np.isnan(bad()))

    def test_event_density(self):
        notes = [
            Note(pitch=80, time=100, duration=100, velocity=80, ttype="tick"),
            Note(pitch=70, time=200, duration=100, velocity=80, ttype="tick"),
            Note(pitch=60, time=300, duration=100, velocity=80, ttype="tick"),
        ]
        score = note_list_to_score(notes, ticks_per_quarter=100)
        expected = 3
        actual = metrics.event_density(score)
        self.assertEqual(expected, actual)
        self.assertEqual(metrics.event_density(Music()), 0)

    def test_tone_spans(self):
        # Test with one big leap
        notes = [
            Note(pitch=80, time=100, duration=100, velocity=80, ttype="tick"),
            Note(pitch=30, time=200, duration=100, velocity=80, ttype="tick"),
            Note(pitch=45, time=300, duration=100, velocity=80, ttype="tick"),
        ]
        expected = 1 / 2
        score = note_list_to_score(notes, ticks_per_quarter=100)
        actual = metrics.tone_spans(score)
        self.assertEqual(expected, actual)
        # Test with no big leaps
        notes = [
            Note(pitch=80, time=100, duration=100, velocity=80, ttype="tick"),
            Note(pitch=70, time=200, duration=100, velocity=80, ttype="tick"),
            Note(pitch=65, time=300, duration=100, velocity=80, ttype="tick"),
        ]
        expected = 0.
        score = note_list_to_score(notes, ticks_per_quarter=100)
        actual = metrics.tone_spans(score)
        self.assertEqual(expected, actual)

    def test_consecutive_pitch_repetitions(self):
        # Test with one repeat note
        notes = [
            Note(pitch=80, time=100, duration=100, velocity=80, ttype="tick"),
            Note(pitch=80, time=200, duration=100, velocity=80, ttype="tick"),
            Note(pitch=75, time=300, duration=100, velocity=80, ttype="tick"),
        ]
        score = note_list_to_score(notes, ticks_per_quarter=100)
        expected = 1 / 2
        actual = metrics.consecutive_pitch_repetitions(score)
        self.assertEqual(expected, actual)
        # Test with no repeat notes
        notes = [
            Note(pitch=80, time=100, duration=100, velocity=80, ttype="tick"),
            Note(pitch=70, time=200, duration=100, velocity=80, ttype="tick"),
            Note(pitch=75, time=300, duration=100, velocity=80, ttype="tick"),
        ]
        score = note_list_to_score(notes, ticks_per_quarter=100)
        expected = 0.
        actual = metrics.consecutive_pitch_repetitions(score)
        self.assertEqual(expected, actual)
        # Test with all repeat notes
        notes = [
            Note(pitch=70, time=100, duration=100, velocity=80, ttype="tick"),
            Note(pitch=70, time=200, duration=100, velocity=80, ttype="tick"),
            Note(pitch=70, time=300, duration=100, velocity=80, ttype="tick"),
        ]
        score = note_list_to_score(notes, ticks_per_quarter=100)
        expected = 1.
        actual = metrics.consecutive_pitch_repetitions(score)
        self.assertEqual(expected, actual)

    def test_consecutive_pitch_class_repetitions(self):
        # Test with one repeat note
        notes = [
            Note(pitch=80, time=100, duration=100, velocity=80, ttype="tick"),
            Note(pitch=68, time=200, duration=100, velocity=80, ttype="tick"),
            Note(pitch=75, time=300, duration=100, velocity=80, ttype="tick"),
        ]
        score = note_list_to_score(notes, ticks_per_quarter=100)
        expected = 1 / 2
        actual = metrics.consecutive_pitch_class_repetitions(score)
        self.assertEqual(expected, actual)
        # Test with no repeat notes
        notes = [
            Note(pitch=80, time=100, duration=100, velocity=80, ttype="tick"),
            Note(pitch=70, time=200, duration=100, velocity=80, ttype="tick"),
            Note(pitch=75, time=300, duration=100, velocity=80, ttype="tick"),
        ]
        score = note_list_to_score(notes, ticks_per_quarter=100)
        expected = 0.
        actual = metrics.consecutive_pitch_class_repetitions(score)
        self.assertEqual(expected, actual)
        # Test with all repeat notes
        notes = [
            Note(pitch=80, time=100, duration=100, velocity=80, ttype="tick"),
            Note(pitch=68, time=200, duration=100, velocity=80, ttype="tick"),
            Note(pitch=56, time=300, duration=100, velocity=80, ttype="tick"),
            Note(pitch=56, time=400, duration=100, velocity=80, ttype="tick"),
            Note(pitch=68, time=500, duration=100, velocity=80, ttype="tick"),
        ]
        score = note_list_to_score(notes, ticks_per_quarter=100)
        expected = 1.
        actual = metrics.consecutive_pitch_class_repetitions(score)
        self.assertEqual(expected, actual)

    def test_rhythmic_variation(self):
        tc = TokenizerConfig(beat_res={(0, 1): 4})  # gives us 4 time-shift tokens
        tok = MIDILike(tc)
        # Should return 0 when no notes passed
        self.assertEqual(metrics.rhythmic_variation([], tok), 0.)
        # Test with a sequence using one rhythm token
        seq = [0, 2, 5, 212]
        expected = 1 / 4
        actual = metrics.rhythmic_variation(seq, tok)
        self.assertEqual(expected, actual)
        # Test with a sequence using two rhythm tokens
        seq = [1, 3, 3, 3, 212, 213]
        expected = 1 / 2
        actual = metrics.rhythmic_variation(seq, tok)
        self.assertEqual(expected, actual)
        # Test with a sequence using all of our rhythm tokens
        seq = [1, 5, 6, 2, 5, 50, 212, 213, 214, 216, 215, 211]
        expected = 1.
        actual = metrics.rhythmic_variation(seq, tok)
        self.assertEqual(expected, actual)
        # Should just return NaN if the tokenizer is trained
        tok.train(vocab_size=500, iterator=[])  # will just train immediately
        seq = [0, 1, 2, 3]
        self.assertTrue(np.isnan(metrics.rhythmic_variation(seq, tok)))

    def test_compute_metrics_for_sequence(self):
        tok = MIDILike()
        # Test with a working sequence
        seqs = torch.tensor([
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [0, 0, 0, 0, 0, 1, 5, 5, 5, 5, 5, ],
            [6, 6, 6, 6, 6, 7, 7, 6, 6, 6, 6, ]
        ])
        parsed = metrics.compute_metrics_for_sequence(seqs, tok)
        self.assertTrue(isinstance(parsed, dict))
        for k, v in parsed.items():
            self.assertTrue(isinstance(k, str))
            self.assertTrue(isinstance(v, float))
        # Test with a bad sequence
        seqs = torch.tensor([[]])
        parsed = metrics.compute_metrics_for_sequence(seqs, tok)
        self.assertTrue(isinstance(parsed, dict))
        for k, v in parsed.items():
            self.assertTrue(isinstance(k, str))
            self.assertTrue(np.isnan(v))

    def test_compute_metrics_for_sequences(self):
        tok = MIDILike()
        # Test with a working sequence
        seqs = [
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        ]
        parsed = metrics.compute_metrics_for_sequences(seqs, tok)
        self.assertTrue(isinstance(parsed, dict))
        for k in metrics.ALL_METRIC_NAMES:
            self.assertTrue(f'{k}_mean' in list(parsed.keys()))
            self.assertTrue(f'{k}_std' in list(parsed.keys()))

    def test_aggregate_evaluation_metrics(self):
        mets = [
            {
                "test1": 1.,
                "test2": 2.,
            },
            {
                "test1": 2.,
                "test2": 3.,
            }
        ]
        agg = metrics.aggregate_evaluation_metrics(mets)
        self.assertEqual(agg["test1_mean"], 1.5)
        self.assertEqual(agg["test2_mean"], 2.5)
        self.assertEqual(agg["test1_std"], 0.5)
        self.assertEqual(agg["test2_std"], 0.5)

    def test_accuracy_score(self):
        toke = load_tokenizer()
        # Test with an un-batched dummy tensor
        dummy_labels = torch.tensor([[1, 2, 2, 1]])
        dummy_tensor = torch.tensor([
            [
                [0.5, 0.6, 0.4],  # correct
                [0.3, 0.2, 0.9],  # correct
                [0.5, 0.2, 0.1],  # incorrect
                [0.3, 0.7, 0.2],  # correct
            ]
        ])
        expected_accuracy = 0.75
        actual_accuracy = metrics.accuracy_score(dummy_tensor, dummy_labels, toke).item()
        self.assertEqual(expected_accuracy, actual_accuracy)
        # Test with a batched dummy tensor
        dummy_labels = torch.tensor(
            [
                [1, 2, 2, 1],
                [2, 2, 2, 2]
            ]
        )
        dummy_tensor = torch.tensor([
            [
                [0.5, 0.6, 0.4],  # correct
                [0.3, 0.2, 0.9],  # correct
                [0.5, 0.2, 0.1],  # incorrect
                [0.3, 0.7, 0.2],  # correct
            ],
            [
                [0.5, 0.6, 1.1],  # correct
                [5.5, 0.2, 1.5],  # incorrect
                [5.5, 0.2, 1.3],  # incorrect
                [5.5, 0.7, 1.2],  # incorrect
            ]
        ])
        expected_accuracy = 0.5
        actual_accuracy = metrics.accuracy_score(dummy_tensor, dummy_labels, toke).item()
        self.assertEqual(expected_accuracy, actual_accuracy)

    def test_accuracy_score_bpe(self):
        """Test accuracy score with BPE-encoded tokens"""
        toke = load_tokenizer()
        # Falsify training a BPE tokenizer
        token_mapping = {
            0: [0],
            1: [1, 2],
            2: [1, 3],
            3: [3],
        }
        setattr(toke, "bpe_token_mapping", token_mapping)
        # Test with an un-batched dummy tensor
        dummy_labels = torch.tensor([[1, 2, 3, 0, 0]])  # decodes to [1, 2, 1, 3, 3], ignoring padding tokens
        dummy_tensor = torch.tensor([
            [
                # 0,   1,   2,   3
                [0.0, 0.5, 0.6, 0.4],  # we predict 2 == [1, 3]. we're correct on 1, but not on 2
                [0.0, 0.3, 0.2, 0.9],  # we predict 3 == 3. we're incorrect at IDX 1, but expanded to IDX2, we are
                [0.0, 0.3, 0.7, 0.2],  # we predict 2 == [1, 3]. we're not correct at either index
            ]
        ])
        expected_accuracy = 2 / 5
        actual_accuracy = metrics.accuracy_score(dummy_tensor, dummy_labels, toke).item()
        self.assertTrue(np.isclose(expected_accuracy, actual_accuracy))  # allow for some floating point tolerance
        # Test with a batched dummy tensor
        dummy_labels = torch.tensor(
            [
                [1, 2, 3, 0, 0],  # decodes to [1, 2, 1, 3, 3], again ignoring padding tokens
                [3, 2, 1, 0, 0]  # decodes to [3, 1, 3, 1, 2]
            ]
        )
        dummy_tensor = torch.tensor([
            [
                # 0,   1,   2,   3
                [0.0, 0.5, 0.6, 0.4],  # we predict 2 == [1, 3]. we're correct on 1, but not on 2
                [0.0, 0.3, 0.2, 0.9],  # we predict 3 == 3. we're incorrect at IDX 1, but expanded to IDX2, we are
                [0.0, 0.3, 0.7, 0.2],  # we predict 2 == [1, 3]. we're not correct at either index
            ],
            [
                [0.0, 5.5, 0.6, 1.1],  # we predict 1 == [1, 2] == [1] (after truncating to match labels) == incorrect
                [0.0, 5.5, 0.2, 1.3],  # we predict 1 == [1, 2]. We're correct at index 1, not at index 2.
                [0.0, 5.5, 0.7, 1.2],  # we predict 1 == [1, 2]. We're correct at both indices
            ]
        ])
        expected_accuracy = 1 / 2
        actual_accuracy = metrics.accuracy_score(dummy_tensor, dummy_labels, toke).item()
        self.assertTrue(np.isclose(expected_accuracy, actual_accuracy))


if __name__ == '__main__':
    unittest.main()
