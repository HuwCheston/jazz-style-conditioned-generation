#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for dataloader"""

import os
import unittest
from copy import deepcopy

import torch
from miditok import MIDILike
from tqdm import tqdm

from jazz_style_conditioned_generation import utils
from jazz_style_conditioned_generation.data.dataloader import *
from jazz_style_conditioned_generation.data.tokenizer import (
    add_tempos_to_vocab,
    add_genres_to_vocab,
    add_pianists_to_vocab,
    add_timesignatures_to_vocab,
    train_tokenizer,
    load_tokenizer
)

TEST_RESOURCES = os.path.join(utils.get_project_root(), "tests/test_resources")
TEST_MIDI1 = os.path.join(TEST_RESOURCES, "test_midi1/piano_midi.mid")
TEST_MIDI2 = os.path.join(TEST_RESOURCES, "test_midi2/piano_midi.mid")
TEST_MIDI3 = os.path.join(TEST_RESOURCES, "test_midi_bushgrafts1/piano_midi.mid")

TOKENIZER = MIDILike()

CONDITION_TOKEN_STARTS = ["GENRES", "PIANIST", "TEMPO", "TIMESIGNATURE"]


def prepare_conditioned_tokenizer():
    token_factory = deepcopy(TOKENIZER)
    # Add in all of our tokens to the vocabulary
    add_genres_to_vocab(token_factory)
    add_pianists_to_vocab(token_factory)
    add_tempos_to_vocab(token_factory, (80, 300), 32)
    add_timesignatures_to_vocab(token_factory, [3, 4])
    return token_factory


def decoder(tokenizer, input_ids: list[int]) -> list[str]:
    converted = tokenizer._convert_sequence_to_tokseq(torch.tensor([input_ids]))
    tokenizer._preprocess_tokseq_before_decoding(converted[0])
    return converted[0].tokens


class DataloaderTest(unittest.TestCase):
    def test_add_eos_bos(self):
        ds = DatasetMIDIRandomChunk(
            tokenizer=TOKENIZER,
            files_paths=[TEST_MIDI1],
            max_seq_len=512,
        )
        tokseq = [2, 2, 2, 3, 4, 5]
        actual = ds.add_beginning_and_ending_tokens_to_sequence(tokseq)
        expected = [TOKENIZER["BOS_None"], 2, 2, 2, 3, 4, 5, TOKENIZER["EOS_None"]]
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
        ds = DatasetMIDIRandomChunk(
            tokenizer=TOKENIZER,
            files_paths=[TEST_MIDI1],
            max_seq_len=desired_len,
            chunk_end_overlap=0.
        )
        # Test a few times
        for _ in range(10):
            returned = ds.randomly_slice_sequence(tokseq)
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
        token_factory = prepare_conditioned_tokenizer()
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
        # Input IDs should start with condition tokens, followed by BOS
        #  IDs are sorted in order of GENRE -> PIANIST -> TEMPO -> TIMESIG
        #  GENRE and PIANIST are sorted in DESCENDING weight order, with the track pianist always placed first
        self.assertEqual(input_ids[0], token_factory["GENRES_Caribbean"])  # most strongly weighted genre, = 10
        self.assertEqual(input_ids[1], token_factory["GENRES_HardBop"])
        self.assertEqual(input_ids[2], token_factory["GENRES_PostBop"])
        self.assertEqual(input_ids[3], token_factory["GENRES_StraightAheadJazz"])
        self.assertEqual(input_ids[4], token_factory["GENRES_Fusion"])  # least strongly weighted genre, = 5
        self.assertEqual(input_ids[5], token_factory["PIANIST_KennyBarron"])
        self.assertEqual(input_ids[6], token_factory["TEMPOCUSTOM_300"])  # closest match to our provided tempo
        self.assertEqual(input_ids[7], token_factory["TIMESIGNATURECUSTOM_44"])
        # Targets should be input_ids, shifted to the left
        self.assertEqual(targets[0], token_factory["GENRES_HardBop"])
        self.assertEqual(targets[1], token_factory["GENRES_PostBop"])
        self.assertEqual(targets[2], token_factory["GENRES_StraightAheadJazz"])
        self.assertEqual(targets[3], token_factory["GENRES_Fusion"])  # least strongly weighted genre, = 5
        self.assertEqual(targets[4], token_factory["PIANIST_KennyBarron"])
        self.assertEqual(targets[5], token_factory["TEMPOCUSTOM_300"])  # closest match to our provided tempo
        self.assertEqual(targets[6], token_factory["TIMESIGNATURECUSTOM_44"])
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
        # self.assertEqual(input_ids[1], token_factory["PIANIST_BeegieAdair"])
        # self.assertEqual(targets[0], token_factory["PIANIST_BeegieAdair"])
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
            for t in CONDITION_TOKEN_STARTS:
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
        # Should raise an error if we try and use augmentation
        with self.assertRaises(NotImplementedError):
            _ = DatasetMIDIExhaustive(
                tokenizer=tokenizer,
                files_paths=[TEST_MIDI1],
                do_augmentation=True,
                max_seq_len=100000,
                do_conditioning=False,
            )

    def test_dataset_exhaustive_with_conditioning(self):
        token_factory = prepare_conditioned_tokenizer()
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
        self.assertEqual(input_ids[0], token_factory["GENRES_Caribbean"])  # most strongly weighted genre, = 10
        self.assertEqual(input_ids[1], token_factory["GENRES_HardBop"])
        self.assertEqual(input_ids[2], token_factory["GENRES_PostBop"])
        self.assertEqual(input_ids[3], token_factory["GENRES_StraightAheadJazz"])
        self.assertEqual(input_ids[4], token_factory["GENRES_Fusion"])  # least strongly weighted genre, = 5
        self.assertEqual(input_ids[5], token_factory["PIANIST_KennyBarron"])
        self.assertEqual(input_ids[6], token_factory["TEMPOCUSTOM_300"])  # closest match to our provided tempo
        self.assertEqual(input_ids[7], token_factory["TIMESIGNATURECUSTOM_44"])
        self.assertEqual(input_ids[8], token_factory["BOS_None"])
        # Targets should be input_ids, shifted to the left
        self.assertEqual(targets[0], token_factory["GENRES_HardBop"])
        self.assertEqual(targets[1], token_factory["GENRES_PostBop"])
        self.assertEqual(targets[2], token_factory["GENRES_StraightAheadJazz"])
        self.assertEqual(targets[3], token_factory["GENRES_Fusion"])  # least strongly weighted genre, = 5
        self.assertEqual(targets[4], token_factory["PIANIST_KennyBarron"])
        self.assertEqual(targets[5], token_factory["TEMPOCUSTOM_300"])  # closest match to our provided tempo
        self.assertEqual(targets[6], token_factory["TIMESIGNATURECUSTOM_44"])
        self.assertEqual(targets[7], token_factory["BOS_None"])
        # Should be the desired length
        self.assertEqual(len(input_ids), 100)
        self.assertEqual(len(targets), 100)
        # Testing the final chunk
        final_item = ds.__getitem__(len(ds) - 1)
        input_ids, targets = final_item["input_ids"].tolist(), final_item["labels"].tolist()
        self.assertEqual(input_ids[0], token_factory["GENRES_Caribbean"])  # most strongly weighted genre
        self.assertEqual(input_ids[7], token_factory["TIMESIGNATURECUSTOM_44"])
        self.assertNotEquals(input_ids[8], token_factory["BOS_None"])  # should NOT have the BOS token here
        # Targets should be input_ids, shifted to the left
        self.assertEqual(targets[0], token_factory["GENRES_HardBop"])
        self.assertEqual(targets[6], token_factory["TIMESIGNATURECUSTOM_44"])
        self.assertNotEquals(targets[7], token_factory["BOS_None"])  # should NOT have the BOS token here
        # and end with the EOS token, after padding is removed
        input_ids_no_pad = [i for i in input_ids if i != token_factory["PAD_None"]]
        self.assertEqual(input_ids_no_pad[-1], token_factory["EOS_None"])

    def test_dataset_conditioning_with_trained_tokenizer(self):
        token_factory = prepare_conditioned_tokenizer()
        train_tokenizer(token_factory, [TEST_MIDI1, TEST_MIDI2, TEST_MIDI3], vocab_size=1000, training_method="BPE")
        # Create the dataset
        ds = DatasetMIDIExhaustive(
            tokenizer=token_factory,
            files_paths=[TEST_MIDI2],
            do_augmentation=False,
            max_seq_len=100,
            do_conditioning=True,
        )
        item = ds.__getitem__(0)
        input_ids, targets = item["input_ids"].tolist(), item["labels"].tolist()
        # We have to do something slightly different to decode the input_ids when using a trained tokenizer
        input_tokens = decoder(token_factory, input_ids)
        target_tokens = decoder(token_factory, targets)
        # Now we can do our checking
        self.assertEqual(input_tokens[0], "GENRES_StraightAheadJazz")  # associated with PIANIST
        # self.assertEqual(input_tokens[1], "PIANIST_BeegieAdair")
        # self.assertEqual(target_tokens[0], "PIANIST_BeegieAdair")
        # With training the token IDs, the length of the decoded tokens > the encoded token IDs
        self.assertGreater(len(input_tokens), len(input_ids))
        self.assertGreater(len(target_tokens), len(targets))

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
            with self.assertRaises(ValueError):
                ds_init.__getitem__(0)

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
        ds = DatasetMIDIRandomChunk(
            tokenizer=TOKENIZER,
            files_paths=[TEST_MIDI1],
            max_seq_len=desired_len,
            chunk_end_overlap=end_overlap
        )
        # Test a few times
        for i in range(10):
            out = ds.randomly_slice_sequence(seq)
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

    def test_add_condition_tokens_to_sequence(self):
        ds = DatasetMIDIRandomChunk(
            tokenizer=TOKENIZER,
            files_paths=[TEST_MIDI1],
            max_seq_len=6,
        )
        dummy = [1, 3, 3, 3, 5, 6]
        condition_tokens = [100, 200, 300]
        # Condition tokens should be added before the start of the sequence, and it should be truncated to fit
        expected_inputs = [100, 200, 300, 1, 3, 3]
        expected_targets = [200, 300, 1, 3, 3, 3]
        actual_inputs, actual_targets = ds.add_condition_tokens_to_input(dummy, condition_tokens)
        self.assertEqual(expected_inputs, actual_inputs)
        self.assertEqual(expected_targets, actual_targets)
        # Testing with adding no condition tokens
        condition_tokens = []
        self.assertRaises(AssertionError, ds.add_condition_tokens_to_input, dummy, condition_tokens)

    @unittest.skipIf(os.getenv("REMOTE") == "true", "Skipping test on GitHub Actions")
    def test_conditioning_full_dataset(self):
        """Test our conditioning with a large number of tracks (will be skipped on remote)"""

        def runner(tokenizer):
            # Create the dataset
            ds = DatasetMIDIRandomChunk(
                tokenizer=tokenizer,
                files_paths=midi_fps,
                max_seq_len=utils.MAX_SEQUENCE_LENGTH,
                do_augmentation=False,
                do_conditioning=True
            )
            # Get the list of condition tokens from our tokenizer
            all_condition_tokens = [i for i in tokenizer.vocab.keys() if i.startswith(tuple(CONDITION_TOKEN_STARTS))]
            # Iterate over every track in the dataset
            for i in tqdm(range(len(ds)), desc="Checking overlap between track and condition tokens..."):
                # Get the input IDs and targets
                input_ids, targets, tempo_shift = ds.load_file(ds.files_paths[i])
                condition_idxs = ds.get_conditioning_tokens(ds.metadata_paths[i], tempo_shift)
                # There should be no overlap between the track IDs and the condition tokens
                self.assertTrue(set(condition_idxs).isdisjoint(set(input_ids)))
                self.assertTrue(set(condition_idxs).isdisjoint(set(targets)))
                # Iterate over "track" tokens (input IDs, targets)
                for tokseq in [input_ids, targets]:
                    decoded = decoder(tokenizer, tokseq)
                    for decoded_token in decoded:
                        # When decoded, input tokens should not be included in our list of condition tokens
                        self.assertFalse(decoded_token in all_condition_tokens)
                        # When decoded, input tokens should NOT start with condition token patterns
                        self.assertFalse(decoded_token.startswith(tuple(CONDITION_TOKEN_STARTS)))
                # Iterate over "condition" tokens
                decoded_condition_tokens = decoder(tokenizer, condition_idxs)
                for decoded_token in decoded_condition_tokens:
                    # When decoded, condition tokens SHOULD be in our list of all condition tokens
                    self.assertTrue(decoded_token in all_condition_tokens)
                    # When decoded, condition tokens SHOULD start with one of our condition token patterns
                    self.assertTrue(decoded_token.startswith(tuple(CONDITION_TOKEN_STARTS)))
                # Decoding the input IDs into a score should give the same results as decoding the input ids + condition
                inputs_decoded = tokenizer.decode(torch.tensor([input_ids]))
                inputs_conditions_decoded = tokenizer.decode(torch.tensor([condition_idxs + input_ids]))
                self.assertEqual(inputs_decoded, inputs_conditions_decoded)
                self.assertEqual(inputs_decoded.tracks, inputs_conditions_decoded.tracks)
                self.assertEqual(inputs_decoded.tracks[0].notes, inputs_conditions_decoded.tracks[0].notes)
                self.assertEqual(len(inputs_decoded.tracks[0].notes), len(inputs_conditions_decoded.tracks[0].notes))

        # Get a large number of tracks + equivalent metadata files
        idx = int(utils.now()[-1]) * 100  # bootleg random index, should operate independently of our set seed
        midi_fps = utils.get_data_files_with_ext("data/raw", "**/*.mid")[idx: idx + 500]
        # Create a tokenizer
        tok = load_tokenizer(tokenizer_str="midilike", )
        add_tempos_to_vocab(tok, (80, 300), 32)
        add_timesignatures_to_vocab(tok, [3, 4])
        add_pianists_to_vocab(tok)
        add_genres_to_vocab(tok)
        # FIRST: we test without training the tokenizer
        runner(tok)
        # SECOND: we test WITH training the tokenizer
        train_tokenizer(tok, midi_fps, vocab_size=1000, training_method="BPE")
        runner(tok)

    def test_adding_condition_tokens_does_not_change_score(self):
        """A score created using raw inputs should be identical to a score created after adding conditions to inputs"""

        def runner(token_factory):
            ds = DatasetMIDIRandomChunk(
                tokenizer=token_factory,
                files_paths=[TEST_MIDI1, TEST_MIDI2, TEST_MIDI3],
                max_seq_len=utils.MAX_SEQUENCE_LENGTH,
                do_augmentation=False,
                do_conditioning=True
            )
            for idx in range(len(ds)):
                # Get the input IDs and targets for the item
                input_ids, targets, tempo_shift = ds.load_file(ds.files_paths[idx])
                input_ids_tensor = torch.tensor([input_ids])  # stack into a tensor for decoding
                # Decode the input IDs into a score
                inputs_decoded = tokenizer.decode(input_ids_tensor)
                # Get the condition tokens for the item and combine with the input ids
                condition_idxs = ds.get_conditioning_tokens(ds.metadata_paths[idx], tempo_shift)
                inputs_with_conditions_tensor = torch.tensor([condition_idxs + input_ids])
                self.assertGreaterEqual(inputs_with_conditions_tensor.size(1), input_ids_tensor.size(1))
                # Decode the condition tokens + input IDs into a score
                inputs_with_conditions_decoded = tokenizer.decode(inputs_with_conditions_tensor)
                # Both scores should be identical
                self.assertEqual(inputs_decoded, inputs_with_conditions_decoded)
                # Both scores should have identical notes, tracks, tempos, and time signatures
                self.assertEqual(inputs_decoded.tracks[0].notes, inputs_with_conditions_decoded.tracks[0].notes)
                self.assertEqual(inputs_decoded.tracks, inputs_with_conditions_decoded.tracks)
                self.assertEqual(inputs_decoded.tempos, inputs_with_conditions_decoded.tempos)
                self.assertEqual(inputs_decoded.time_signatures, inputs_with_conditions_decoded.time_signatures)
                self.assertEqual(inputs_decoded.ticks_per_quarter, inputs_with_conditions_decoded.ticks_per_quarter)

        # FIRST, we test without training the tokenizer
        tokenizer = prepare_conditioned_tokenizer()
        runner(tokenizer)
        # SECOND, we train the tokenizer and test again
        train_tokenizer(tokenizer, [TEST_MIDI1], vocab_size=1000, training_method="BPE")
        runner(tokenizer)


if __name__ == '__main__':
    utils.seed_everything(utils.SEED)
    unittest.main()
