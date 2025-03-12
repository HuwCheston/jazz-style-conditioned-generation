#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for dataloader"""

import os
import unittest
from copy import deepcopy

from miditok import MIDILike

from jazz_style_conditioned_generation import utils
from jazz_style_conditioned_generation.data.dataloader import *

TEST_RESOURCES = os.path.join(utils.get_project_root(), "tests/test_resources")
TEST_MIDI = os.path.join(TEST_RESOURCES, "test_midi1/piano_midi.mid")

TOKENIZER = MIDILike()


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
        # Test a few times
        for _ in range(10):
            returned = randomly_slice_sequence(tokseq, desired_len, end_overlap=0.)
            self.assertTrue(len(returned) < len(tokseq))
            self.assertEqual(len(returned), desired_len)

    def test_dataset_random_chunk_getitem(self):
        ds = DatasetMIDIRandomChunk(
            tokenizer=TOKENIZER,
            files_paths=[TEST_MIDI],
            do_augmentation=False,
            max_seq_len=512,
            do_conditioning=False,
            chunk_end_overlap=0.
        )
        gotitem = ds.__getitem__(0)
        input_ids, targets = gotitem["input_ids"], gotitem["labels"]
        # Input IDs and labels should be the target length
        self.assertEqual(input_ids.size(0), 512)
        self.assertEqual(targets.size(0), 512)
        # Labels should be the input IDs shifted by one
        self.assertEqual(input_ids.tolist()[1:], targets.tolist()[:-1])
        # With a chunk_end_overlap of 0., we shouldn't have any padding tokens in the output
        self.assertFalse(True in gotitem["attention_mask"])

    def test_dataset_random_chunk(self):
        max_seq_length = 10
        ds = DatasetMIDIRandomChunk(
            tokenizer=TOKENIZER,
            files_paths=[TEST_MIDI],
            do_augmentation=False,
            max_seq_len=10,
            do_conditioning=False,
            chunk_end_overlap=0.
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

    def test_dataset_random_chunk_with_conditioning(self):
        token_factory = deepcopy(TOKENIZER)
        token_factory.add_to_vocab("PIANIST_KennyBarron")
        token_factory.add_to_vocab("GENRES_HardBop")
        ds = DatasetMIDIRandomChunk(
            tokenizer=token_factory,
            files_paths=[TEST_MIDI],
            do_augmentation=False,
            max_seq_len=10,
            do_conditioning=True,
            chunk_end_overlap=0.,
            condition_mapping={
                "pianist": {"Kenny Barron": "PIANIST_KennyBarron"},
                "genres": {"Hard Bop": "GENRES_HardBop"},
            }
        )
        self.assertEqual(len(ds), 1)
        item = ds.__getitem__(0)
        input_ids, targets = item["input_ids"].tolist(), item["labels"].tolist()
        # Input IDs should start with the expected conditioning tokens
        self.assertEqual(input_ids[0], token_factory["GENRES_HardBop"])
        self.assertEqual(input_ids[1], token_factory["PIANIST_KennyBarron"])
        self.assertEqual(targets[0], token_factory["PIANIST_KennyBarron"])
        # Should be the desired length
        self.assertEqual(len(input_ids), 10)
        self.assertEqual(len(targets), 10)

    def test_dataset_exhaustive(self):
        tokenizer = deepcopy(TOKENIZER)
        # Test with a low max_seq_len (== lots of chunks)
        ds_small = DatasetMIDIExhaustive(
            tokenizer=tokenizer,
            files_paths=[TEST_MIDI],
            do_augmentation=False,
            max_seq_len=10,
            do_conditioning=False
        )
        self.assertTrue(len(ds_small) > 1)
        # Test with a high max_seq_len (== few chunks)
        ds_big = DatasetMIDIExhaustive(
            tokenizer=tokenizer,
            files_paths=[TEST_MIDI],
            do_augmentation=False,
            max_seq_len=100000,
            do_conditioning=False
        )
        self.assertTrue(len(ds_big) == 1)

        for ds in [ds_small, ds_big]:
            # First chunk should start with BOS
            first_chunk = ds.__getitem__(0)
            actual_inputs, _ = first_chunk["input_ids"], first_chunk["labels"]
            self.assertTrue(actual_inputs[0].item() == tokenizer["BOS_None"])
            # Last chunk should end with EOS
            last_chunk = ds.__getitem__(-1)
            actual_inputs, _ = last_chunk["input_ids"], last_chunk["labels"]
            actual_ids = [i_ for i_ in actual_inputs.tolist() if i_ != tokenizer["PAD_None"]]
            self.assertTrue(actual_ids[-1] == tokenizer["EOS_None"])

    def test_dataset_exhaustive_with_conditioning(self):
        token_factory = deepcopy(TOKENIZER)
        token_factory.add_to_vocab("PIANIST_KennyBarron")
        token_factory.add_to_vocab("GENRES_HardBop")
        ds = DatasetMIDIExhaustive(
            tokenizer=token_factory,
            files_paths=[TEST_MIDI],
            do_augmentation=False,
            max_seq_len=100,
            do_conditioning=True,
            condition_mapping={
                "pianist": {"Kenny Barron": "PIANIST_KennyBarron"},
                "genres": {"Hard Bop": "GENRES_HardBop"},
            }
        )
        item = ds.__getitem__(0)
        input_ids, targets = item["input_ids"].tolist(), item["labels"].tolist()
        # Input IDs should start with condition tokens, followed by BOS
        self.assertEqual(input_ids[0], token_factory["GENRES_HardBop"])
        self.assertEqual(input_ids[1], token_factory["PIANIST_KennyBarron"])
        self.assertEqual(input_ids[2], token_factory["BOS_None"])
        self.assertEqual(targets[0], token_factory["PIANIST_KennyBarron"])
        self.assertEqual(targets[1], token_factory["BOS_None"])
        # Should be the desired length
        self.assertEqual(len(input_ids), 100)
        self.assertEqual(len(targets), 100)
        # Testing the final chunk
        final_item = ds.__getitem__(len(ds) - 1)
        input_ids, targets = final_item["input_ids"].tolist(), final_item["labels"].tolist()
        # Should start with the condition tokens now
        self.assertEqual(input_ids[0], token_factory["GENRES_HardBop"])
        self.assertEqual(input_ids[1], token_factory["PIANIST_KennyBarron"])
        self.assertEqual(targets[0], token_factory["PIANIST_KennyBarron"])
        # and end with the EOS token, after padding is removed
        input_ids_no_pad = [i for i in input_ids if i != token_factory["PAD_None"]]
        self.assertEqual(input_ids_no_pad[-1], token_factory["EOS_None"])

    def test_dataset_with_conditioning_merges(self):
        token_factory = deepcopy(TOKENIZER)
        # This track has the tag "Calypso". We are expecting this to be merged with the tag "GENRES_Caribbean"
        token_factory.add_to_vocab("PIANIST_KennyBarron")
        token_factory.add_to_vocab("GENRES_HardBop")
        token_factory.add_to_vocab("GENRES_Caribbean")
        ds = DatasetMIDIExhaustive(
            tokenizer=token_factory,
            files_paths=[TEST_MIDI],
            do_augmentation=False,
            max_seq_len=100,
            do_conditioning=True,
            condition_mapping={
                "pianist": {"Kenny Barron": "PIANIST_KennyBarron"},
                "genres": {"Caribbean": "GENRES_Caribbean", "Hard Bop": "GENRES_HardBop"},
            }
        )
        item = ds.__getitem__(0)
        input_ids, targets = item["input_ids"].tolist(), item["labels"].tolist()
        # Input IDs should start with BOS, followed by the expected conditioning tokens
        self.assertEqual(input_ids[0], token_factory["GENRES_Caribbean"])
        self.assertEqual(input_ids[1], token_factory["GENRES_HardBop"])
        self.assertEqual(input_ids[2], token_factory["PIANIST_KennyBarron"])
        self.assertEqual(input_ids[3], token_factory["BOS_None"])
        self.assertEqual(targets[0], token_factory["GENRES_HardBop"])
        self.assertEqual(targets[1], token_factory["PIANIST_KennyBarron"])
        self.assertEqual(targets[2], token_factory["BOS_None"])
        # Should be the desired length
        self.assertEqual(len(input_ids), 100)
        self.assertEqual(len(targets), 100)
        # Testing the final chunk
        final_item = ds.__getitem__(len(ds) - 1)
        input_ids, targets = final_item["input_ids"].tolist(), final_item["labels"].tolist()
        # Should start with the condition tokens
        self.assertEqual(input_ids[0], token_factory["GENRES_Caribbean"])
        self.assertEqual(input_ids[1], token_factory["GENRES_HardBop"])
        self.assertEqual(input_ids[2], token_factory["PIANIST_KennyBarron"])
        self.assertEqual(targets[0], token_factory["GENRES_HardBop"])
        self.assertEqual(targets[1], token_factory["PIANIST_KennyBarron"])
        # and end with the EOS token, after padding is removed
        input_ids_no_pad = [i for i in input_ids if i != token_factory["PAD_None"]]
        self.assertEqual(input_ids_no_pad[-1], token_factory["EOS_None"])

    def test_non_existing_filepath(self):
        tokenizer = deepcopy(TOKENIZER)
        # This API is the same for both datasets
        for ds in [DatasetMIDIExhaustive, DatasetMIDIRandomChunk]:
            ds_init = ds(
                tokenizer=tokenizer,
                files_paths=[TEST_MIDI],
                do_augmentation=False,
                max_seq_len=10,
                do_conditioning=False
            )
            # Modify the file paths
            ds_init.files_paths = ["a/fake/file"]
            ds_init.chunk_paths_and_idxs = [("a/fake/file", 0)]
            # Values from dictionary should be None
            gotitem = ds_init.__getitem__(0)
            self.assertIsNone(gotitem["input_ids"])
            self.assertIsNone(gotitem["labels"])

    def test_dataset_attention_mask(self):
        # Test exhaustive dataloader
        token_factory = deepcopy(TOKENIZER)
        ds = DatasetMIDIExhaustive(
            tokenizer=token_factory,
            files_paths=[TEST_MIDI],
            do_augmentation=False,
            max_seq_len=100,
            do_conditioning=False
        )
        # First chunk should not have any padding
        first_chunk = ds.__getitem__(0)
        self.assertFalse(True in first_chunk["attention_mask"])
        # Last chunk should have padding
        last_chunk = ds.__getitem__(len(ds) - 1)
        self.assertTrue(True in last_chunk["attention_mask"])

        # Test random chunk dataloader
        ds = DatasetMIDIRandomChunk(
            tokenizer=token_factory,
            files_paths=[TEST_MIDI],
            do_augmentation=False,
            max_seq_len=10,
            do_conditioning=False,
            chunk_end_overlap=0.
        )
        # Random chunks FROM THIS TRACK (WHICH IS LONG) should not have any padding
        #  NB., if we had a track which was very short, we would expect some padding here
        first_chunk = ds.__getitem__(0)
        self.assertFalse(True in first_chunk["attention_mask"])

    def test_random_chunk_end_overlap(self):
        # With a sequence of 10 items and an end overlap of 0.9, our sequence can vary in length from 1:8 items
        seq = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        desired_len = 8
        end_overlap = 0.9
        # Test a few times
        for i in range(10):
            out = randomly_slice_sequence(seq, desired_len, end_overlap)
            self.assertTrue(1 <= len(out) <= desired_len)
        # Test this out with a dataloader
        token_factory = deepcopy(TOKENIZER)
        ds = DatasetMIDIRandomChunk(
            tokenizer=token_factory,
            files_paths=[TEST_MIDI],
            do_augmentation=False,
            max_seq_len=100000,
            do_conditioning=False,
            chunk_end_overlap=0.9
        )
        # We've constructed the loader such that we'll expect padding
        first_chunk = ds.__getitem__(0)
        self.assertTrue(token_factory["PAD_None"] in first_chunk["input_ids"])
        self.assertTrue(True in first_chunk["attention_mask"])

    def test_attention_mask(self):
        # Testing with padding at end
        tokseq = [1, 2, 3, 4, 82, 234, 62, 0, 0, 0]
        expected = [False, False, False, False, False, False, False, True, True, True]
        actual = create_padding_mask(tokseq, pad_token_id=0)
        self.assertEqual(expected, actual.tolist())
        # Testing with no padding
        tokseq = [55, 55, 55, 55, 55, 55, 55, 66, 66, 66, ]
        expected = [False, False, False, False, False, False, False, False, False, False, ]
        actual = create_padding_mask(tokseq, pad_token_id=0)
        self.assertEqual(expected, actual.tolist())


if __name__ == '__main__':
    utils.seed_everything(utils.SEED)
    unittest.main()
