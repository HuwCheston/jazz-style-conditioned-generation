#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for dataloader"""

import os
import unittest

from miditok import REMI
from symusic import Score

from jazz_style_conditioned_generation import utils
from jazz_style_conditioned_generation.data.dataloader import *

TEST_MIDI = os.path.join(utils.get_project_root(), "tests/test_resources/test_midi1.mid")


class DataloaderTest(unittest.TestCase):
    def test_add_eos_bos(self):
        tokseq = [2, 2, 2, 3, 4, 5]
        actual = add_beginning_and_ending_tokens_to_sequence(tokseq, 0, 1)
        expected = [0, 2, 2, 2, 3, 4, 5, 1]
        self.assertEqual(actual, expected)

    def test_pad_sequence(self):
        # Test right padding
        tokseq = [2, 2, 2, 3, 4, 5]
        expected = [2, 2, 2, 3, 4, 5, 0, 0, 0, 0]
        actual = pad_sequence(tokseq, desired_len=10, pad_token_id=0)
        self.assertEqual(actual, expected)
        # Test left padding
        expected = [0, 0, 0, 0, 2, 2, 2, 3, 4, 5]
        actual = pad_sequence(tokseq, desired_len=10, pad_token_id=0, right_pad=False)
        self.assertEqual(actual, expected)

    def test_randomly_slice(self):
        tokseq = [2, 2, 2, 3, 4, 5, 7, 6, 4, 3, 8, ]
        desired_len = 5
        returned = randomly_slice_sequence(tokseq, desired_len)
        self.assertTrue(len(returned) < len(tokseq))
        self.assertEqual(len(returned), desired_len)

    def test_get_pitch_range(self):
        score = Score(TEST_MIDI)
        expected_min_pitch, expected_max_pitch = 34, 98
        actual_min_pitch, actual_max_pitch = utils.get_pitch_range(score)
        self.assertEqual(expected_min_pitch, actual_min_pitch)
        self.assertEqual(expected_max_pitch, actual_max_pitch)

    def test_get_pitch_augment_value(self):
        # Test with a real midi clip
        score = Score(TEST_MIDI)
        expected_max_augment = 3
        actual_augment = get_pitch_augmentation_value(score, PITCH_AUGMENT_RANGE)
        self.assertTrue(abs(actual_augment) <= expected_max_augment)
        # TODO: test with a midi clip that exceeds boundaries

    def test_data_augmentation(self):
        score = Score(TEST_MIDI)
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

    def test_dataset_random_chunk_getitem(self):
        ds = DatasetMIDIRandomChunk(
            tokenizer=REMI(),
            files_paths=[TEST_MIDI],
            do_augmentation=False,
            max_seq_len=512
        )
        gotitem = ds.__getitem__(0)
        input_ids, targets = gotitem["input_ids"], gotitem["labels"]
        # Input IDs and labels should be the target length
        self.assertEqual(input_ids.size(0), 512)
        self.assertEqual(targets.size(0), 512)
        # Labels should be the input IDs shifted by one
        self.assertEqual(input_ids.tolist()[1:], targets.tolist()[:-1])

    def test_dataset_random_chunk(self):
        max_seq_length = 10
        ds = DatasetMIDIRandomChunk(
            tokenizer=REMI(),
            files_paths=[TEST_MIDI],
            do_augmentation=False,
            max_seq_len=10
        )
        self.assertEqual(len(ds), 1)

        # Test chunking a sequence that is too short: should get padded
        seq = [3, 4, 5, 6, 7, 8, 9]
        expected_inputs = [3, 4, 5, 6, 7, 8, 9, 0, 0, 0]
        expected_targets = [4, 5, 6, 7, 8, 9, 0, 0, 0, 0]
        actual_inputs, actual_targets = ds.chunk_sequence(seq)
        self.assertEqual(actual_inputs, expected_inputs)
        self.assertEqual(actual_targets, expected_targets)
        self.assertEqual(len(actual_targets), max_seq_length)
        self.assertEqual(len(actual_inputs), max_seq_length)
        self.assertEqual(len(actual_targets), len(actual_inputs))
        self.assertEqual(actual_inputs[1:], actual_targets[:-1])

        # Test chunking a sequence that is too long: should be truncated
        seq = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        actual_inputs, actual_targets = ds.chunk_sequence(seq)
        self.assertEqual(len(actual_targets), max_seq_length)
        self.assertEqual(len(actual_inputs), max_seq_length)
        self.assertEqual(len(actual_targets), len(actual_inputs))
        self.assertEqual(actual_inputs[1:], actual_targets[:-1])

        # Test chunking a sequence that is exactly the target length: should stay the same
        seq = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        self.assertEqual(len(seq), max_seq_length)
        actual_inputs, actual_targets = ds.chunk_sequence(seq)
        self.assertEqual(len(actual_targets), max_seq_length)
        self.assertEqual(len(actual_inputs), max_seq_length)
        self.assertEqual(actual_inputs, seq)
        self.assertEqual(actual_targets, [4, 5, 6, 7, 8, 9, 10, 11, 12, 0])

    def test_dataset_exhaustive(self):
        tokenizer = REMI()
        # Test with a low max_seq_len (== lots of chunks)
        ds_small = DatasetMIDIExhaustive(
            tokenizer=tokenizer,
            files_paths=[TEST_MIDI],
            do_augmentation=False,
            max_seq_len=10
        )
        self.assertTrue(len(ds_small) > 1)
        # Test with a high max_seq_len (== few chunks)
        ds_big = DatasetMIDIExhaustive(
            tokenizer=REMI(),
            files_paths=[TEST_MIDI],
            do_augmentation=False,
            max_seq_len=100000
        )
        self.assertTrue(len(ds_big) == 1)

        for ds in [ds_small, ds_big]:
            # First chunk should start with BOS
            first_chunk = ds.__getitem__(0)
            actual_inputs, _ = first_chunk["input_ids"], first_chunk["labels"]
            self.assertTrue(actual_inputs[0].item() == tokenizer["BOS_None"])
            # Last chunk should end with EOS
            first_chunk = ds.__getitem__(len(ds) - 1)
            actual_inputs, _ = first_chunk["input_ids"], first_chunk["labels"]
            actual_ids = [i_ for i_ in actual_inputs.tolist() if i_ != tokenizer["PAD_None"]]
            self.assertTrue(actual_ids[-1] == tokenizer["EOS_None"])

    def test_non_existing_filepath(self):
        tokenizer = REMI()
        # This API is the same for both datasets
        for ds in [DatasetMIDIExhaustive, DatasetMIDIRandomChunk]:
            ds_init = ds(
                tokenizer=tokenizer,
                files_paths=[TEST_MIDI],
                do_augmentation=False,
                max_seq_len=10
            )
            # Modify the file paths
            ds_init.files_paths = ["a/fake/file"]
            ds_init.chunk_paths_and_idxs = [("a/fake/file", 0)]
            # Values from dictionary should be None
            gotitem = ds_init.__getitem__(0)
            self.assertIsNone(gotitem["input_ids"])
            self.assertIsNone(gotitem["labels"])


if __name__ == '__main__':
    unittest.main()
