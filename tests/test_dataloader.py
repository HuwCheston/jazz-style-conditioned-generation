#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for dataloader"""

import os
import unittest
from copy import deepcopy

from miditok import MIDILike

from jazz_style_conditioned_generation import utils
from jazz_style_conditioned_generation.data.dataloader import *
from jazz_style_conditioned_generation.data.tokenizer import (
    add_tempos_to_vocab,
    add_genres_to_vocab,
    add_pianists_to_vocab,
    add_timesignatures_to_vocab
)

TEST_RESOURCES = os.path.join(utils.get_project_root(), "tests/test_resources")
TEST_MIDI1 = os.path.join(TEST_RESOURCES, "test_midi1/piano_midi.mid")
TEST_MIDI2 = os.path.join(TEST_RESOURCES, "test_midi2/piano_midi.mid")
TEST_MIDI3 = os.path.join(TEST_RESOURCES, "test_midi_bushgrafts1/piano_midi.mid")

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
            files_paths=[TEST_MIDI1],
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
            files_paths=[TEST_MIDI1],
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
        # Add in all of our tokens to the vocabulary
        testers = [TEST_MIDI1, TEST_MIDI2, TEST_MIDI3]
        metadata_fps = [tm.replace("piano_midi.mid", "metadata_tivo.json") for tm in testers]
        add_genres_to_vocab(token_factory, metadata_fps)
        add_pianists_to_vocab(token_factory, metadata_fps)
        add_tempos_to_vocab(token_factory, (80, 300), 32)
        add_timesignatures_to_vocab(token_factory, [3, 4])

        # Create the dataset with MIDI file 1
        ds = DatasetMIDIRandomChunk(
            tokenizer=token_factory,
            files_paths=[TEST_MIDI1],  # this track has a bpm of 297-ish and a time signature of 4/4
            do_augmentation=False,
            max_seq_len=10,
            do_conditioning=True,
            chunk_end_overlap=0.,
        )
        self.assertEqual(len(ds), 1)
        item = ds.__getitem__(0)
        input_ids, targets = item["input_ids"].tolist(), item["labels"].tolist()
        # Input IDs should start with the expected conditioning tokens
        #  IDs are sorted in order of GENRE -> PIANIST -> TEMPO -> TIMESIG
        #  GENRE and PIANIST are sorted in DESCENDING weight order, with the track pianist always placed first
        self.assertEqual(input_ids[0], token_factory["GENRES_Caribbean"])  # most strongly weighted genre
        self.assertEqual(input_ids[1], token_factory["GENRES_HardBop"])
        self.assertEqual(input_ids[2], token_factory["GENRES_PostBop"])  # least strongly weighted genre
        self.assertEqual(input_ids[3], token_factory["PIANIST_KennyBarron"])
        self.assertEqual(input_ids[4], token_factory["TEMPOCUSTOM_300"])  # closest match to our provided tempo
        self.assertEqual(input_ids[5], token_factory["TIMESIGNATURECUSTOM_44"])
        # Targets are just the inputs shifted by one
        self.assertEqual(targets[0], token_factory["GENRES_HardBop"])
        self.assertEqual(targets[1], token_factory["GENRES_PostBop"])
        self.assertEqual(targets[2], token_factory["PIANIST_KennyBarron"])
        self.assertEqual(targets[3], token_factory["TEMPOCUSTOM_300"])
        self.assertEqual(targets[4], token_factory["TIMESIGNATURECUSTOM_44"])
        # Should be the desired length
        self.assertEqual(len(input_ids), 10)
        self.assertEqual(len(targets), 10)

        # Create the dataset with MIDI file 1
        ds = DatasetMIDIRandomChunk(
            tokenizer=token_factory,
            files_paths=[TEST_MIDI2],  # this track has a bpm of 297-ish and a time signature of 4/4
            do_augmentation=False,
            max_seq_len=10,
            do_conditioning=True,
            chunk_end_overlap=0.,
        )
        self.assertEqual(len(ds), 1)
        item = ds.__getitem__(0)
        input_ids, targets = item["input_ids"].tolist(), item["labels"].tolist()
        # This track has one GENRE and one PIANIST token. The GENRE token is associated with the PIANIST, not the track
        self.assertEqual(input_ids[0], token_factory["GENRES_StraightAheadJazz"])  # associated with PIANIST
        self.assertEqual(input_ids[1], token_factory["PIANIST_BeegieAdair"])
        self.assertEqual(targets[0], token_factory["PIANIST_BeegieAdair"])
        # We should not have any tempo or time signature tokens for this track
        for tok in input_ids:
            for t in ["TEMPO", "TIMESIGNATURE"]:
                self.assertFalse(token_factory[tok].startswith(t))
        # Should be the desired length
        self.assertEqual(len(input_ids), 10)
        self.assertEqual(len(targets), 10)

        # Create a dataset with another MIDI, that should NOT have any genre/pianist information
        ds = DatasetMIDIRandomChunk(
            tokenizer=token_factory,
            files_paths=[TEST_MIDI3],
            do_augmentation=False,
            max_seq_len=10,
            do_conditioning=True,
            chunk_end_overlap=0.,
        )
        self.assertEqual(len(ds), 1)
        item = ds.__getitem__(0)
        input_ids, targets = item["input_ids"].tolist(), item["labels"].tolist()
        # Input IDs should NOT start with any conditioning tokens
        for tok in input_ids:
            for t in ["GENRES", "PIANIST", "TEMPO", "TIMESIGNATURE"]:
                self.assertFalse(token_factory[tok].startswith(t))
        # Should be the desired length
        self.assertEqual(len(input_ids), 10)
        self.assertEqual(len(targets), 10)

    def test_dataset_exhaustive(self):
        tokenizer = deepcopy(TOKENIZER)
        # Test with a low max_seq_len (== lots of chunks)
        ds_small = DatasetMIDIExhaustive(
            tokenizer=tokenizer,
            files_paths=[TEST_MIDI1],
            do_augmentation=False,
            max_seq_len=10,
            do_conditioning=False
        )
        self.assertTrue(len(ds_small) > 1)
        # Test with a high max_seq_len (== few chunks)
        ds_big = DatasetMIDIExhaustive(
            tokenizer=tokenizer,
            files_paths=[TEST_MIDI1],
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
        # Add in all of our tokens to the vocabulary
        metadata_fps = [TEST_MIDI1.replace("piano_midi.mid", "metadata_tivo.json")]
        add_genres_to_vocab(token_factory, metadata_fps)
        add_pianists_to_vocab(token_factory, metadata_fps)
        add_tempos_to_vocab(token_factory, (80, 300), 32)
        add_timesignatures_to_vocab(token_factory, [3, 4])
        # Create the dataset
        ds = DatasetMIDIExhaustive(
            tokenizer=token_factory,
            files_paths=[TEST_MIDI1],
            do_augmentation=False,
            max_seq_len=100,
            do_conditioning=True,
        )
        item = ds.__getitem__(0)
        input_ids, targets = item["input_ids"].tolist(), item["labels"].tolist()
        # Input IDs should start with condition tokens, followed by BOS
        #  IDs are sorted in order of GENRE -> PIANIST -> TEMPO -> TIMESIG
        #  GENRE and PIANIST are sorted in DESCENDING weight order, with the track pianist always placed first
        self.assertEqual(input_ids[0], token_factory["GENRES_Caribbean"])  # most strongly weighted genre
        self.assertEqual(input_ids[1], token_factory["GENRES_HardBop"])
        self.assertEqual(input_ids[2], token_factory["GENRES_PostBop"])  # least strongly weighted genre
        self.assertEqual(input_ids[3], token_factory["PIANIST_KennyBarron"])
        self.assertEqual(input_ids[4], token_factory["TEMPOCUSTOM_300"])  # closest match to our provided tempo
        self.assertEqual(input_ids[5], token_factory["TIMESIGNATURECUSTOM_44"])
        self.assertEqual(input_ids[6], token_factory["BOS_None"])
        # Targets should be input_ids, shifted to the left
        self.assertEqual(targets[0], token_factory["GENRES_HardBop"])
        self.assertEqual(targets[1], token_factory["GENRES_PostBop"])  # least strongly weighted genre
        self.assertEqual(targets[2], token_factory["PIANIST_KennyBarron"])
        self.assertEqual(targets[3], token_factory["TEMPOCUSTOM_300"])  # closest match to our provided tempo
        self.assertEqual(targets[4], token_factory["TIMESIGNATURECUSTOM_44"])
        self.assertEqual(targets[5], token_factory["BOS_None"])
        # Should be the desired length
        self.assertEqual(len(input_ids), 100)
        self.assertEqual(len(targets), 100)
        # Testing the final chunk
        final_item = ds.__getitem__(len(ds) - 1)
        input_ids, targets = final_item["input_ids"].tolist(), final_item["labels"].tolist()
        self.assertEqual(input_ids[0], token_factory["GENRES_Caribbean"])  # most strongly weighted genre
        self.assertEqual(input_ids[5], token_factory["TIMESIGNATURECUSTOM_44"])
        self.assertNotEquals(input_ids[6], token_factory["BOS_None"])  # should NOT have the BOS token here
        # Targets should be input_ids, shifted to the left
        self.assertEqual(targets[0], token_factory["GENRES_HardBop"])
        self.assertEqual(targets[4], token_factory["TIMESIGNATURECUSTOM_44"])
        self.assertNotEquals(targets[5], token_factory["BOS_None"])
        # and end with the EOS token, after padding is removed
        input_ids_no_pad = [i for i in input_ids if i != token_factory["PAD_None"]]
        self.assertEqual(input_ids_no_pad[-1], token_factory["EOS_None"])

    def test_non_existing_filepath(self):
        tokenizer = deepcopy(TOKENIZER)
        # This API is the same for both datasets
        for ds in [DatasetMIDIExhaustive, DatasetMIDIRandomChunk]:
            ds_init = ds(
                tokenizer=tokenizer,
                files_paths=[TEST_MIDI1],
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
            files_paths=[TEST_MIDI1],
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
            files_paths=[TEST_MIDI1],
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
            files_paths=[TEST_MIDI1],
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
