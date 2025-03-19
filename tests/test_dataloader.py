#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for dataloader"""

import os
import random
import unittest
from copy import deepcopy

import torch
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

TOKENIZER = load_tokenizer(tokenizer_str="midilike")
DUMMY_DATASET = DatasetMIDIConditioned(
    tokenizer=TOKENIZER,
    files_paths=[TEST_MIDI1],
    max_seq_len=512,
    do_conditioning=False,
    do_augmentation=False
)

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


def scores_are_identical(score_a, score_b) -> bool:
    return all([
        score_a == score_b,
        score_a.tracks == score_b.tracks,
        score_a.tracks[0].notes == score_b.tracks[0].notes,
        len(score_a.tracks[0].notes) == len(score_b.tracks[0].notes),
        score_a.tempos == score_b.tempos,
        score_a.time_signatures == score_b.time_signatures,
        score_a.ticks_per_quarter == score_b.ticks_per_quarter,
    ])


class DatasetConditionedTest(unittest.TestCase):
    def test_add_eos_bos(self):
        tokseq = [2, 2, 2, 3, 4, 5]
        actual = DUMMY_DATASET.add_beginning_and_ending_tokens_to_sequence(tokseq)
        expected = [TOKENIZER["BOS_None"], 2, 2, 2, 3, 4, 5, TOKENIZER["EOS_None"]]
        self.assertEqual(actual, expected)

    def test_attention_mask(self):
        # First chunk should not have any padding
        first_chunk = DUMMY_DATASET.__getitem__(0)
        self.assertFalse(True in first_chunk["attention_mask"])
        # Last chunk should have padding
        last_chunk = DUMMY_DATASET.__getitem__(len(DUMMY_DATASET) - 1)
        self.assertTrue(True in last_chunk["attention_mask"])

    def test_create_attention_mask(self):
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

    def test_get_conditioning_tokens(self):
        token_factory = prepare_conditioned_tokenizer()
        ds = DatasetMIDIConditioned(
            tokenizer=token_factory,
            files_paths=[TEST_MIDI1, TEST_MIDI2, TEST_MIDI3],
            max_seq_len=512,
            do_conditioning=True,
            do_augmentation=False
        )
        # Test with metadata from first track
        #  GENRE and PIANIST are sorted in DESCENDING weight order, with the track pianist always placed first
        expected_tokens = [
            "GENRES_Caribbean", "GENRES_HardBop", "GENRES_PostBop", "GENRES_StraightAheadJazz", "GENRES_Fusion",
            "PIANIST_KennyBarron", "TEMPOCUSTOM_300", "TIMESIGNATURECUSTOM_44"
        ]
        expected_token_ids = [token_factory[t] for t in expected_tokens]
        actual_token_ids = ds.get_conditioning_tokens(utils.read_json_cached(ds.metadata_paths[0]))
        self.assertTrue(expected_token_ids == actual_token_ids)
        # Test with metadata from second track
        # This track has one GENRE token, associated with the pianist, and two SIMILAR PIANISTS
        expected_tokens = ["GENRES_StraightAheadJazz", "PIANIST_BradMehldau", "PIANIST_KennyDrew"]
        expected_token_ids = [token_factory[t] for t in expected_tokens]
        actual_token_ids = ds.get_conditioning_tokens(utils.read_json_cached(ds.metadata_paths[1]))
        self.assertTrue(expected_token_ids == actual_token_ids)
        # Test with metadata from third track: should not have any conditioning tokens!
        actual_token_ids = ds.get_conditioning_tokens(utils.read_json_cached(ds.metadata_paths[2]))
        self.assertTrue([] == actual_token_ids)

    def test_shift_labels(self):
        expected_input_ids = list(range(0, 512))  # [0, 1, 2, 3, ..., 511]
        expected_targets = list(range(1, 513))  # [1, 2, 3, 4, ..., 512]
        actual_input_ids, actual_targets = DUMMY_DATASET.shift_labels(list(range(0, 513)))  # [0, 1, 2, 3, ..., 512]
        self.assertEqual(expected_input_ids, actual_input_ids)
        self.assertEqual(expected_targets, actual_targets)

    def test_scale_tempo(self):
        # Test with a track that is made shorter: BPM should INCREASE
        expected = 100
        actual = round(DUMMY_DATASET.scale_tempo(90, 0.9))
        self.assertEqual(expected, actual)
        # Test with a track that is made longer, BPM should DECREASE
        expected = 100
        actual = round(DUMMY_DATASET.scale_tempo(110, 1.1))
        self.assertEqual(expected, actual)
        # Test with a track that has no change
        expected = 100
        actual = DUMMY_DATASET.scale_tempo(100, 1.)
        self.assertEqual(expected, actual)

    def test_scale_slice_indices(self):
        slice_start, slice_end = 90, 200
        # We're making the track shorter, therefore the slice should come earlier
        scale = 0.9
        expected = 81, 191
        actual = DUMMY_DATASET.scale_slice_indices(slice_start, slice_end, scale)
        self.assertEqual(expected, actual)
        # We're making the track longer, therefore the slice should come later
        scale = 1.1
        expected = 99, 209
        actual = DUMMY_DATASET.scale_slice_indices(slice_start, slice_end, scale)
        self.assertEqual(expected, actual)
        # We're keeping the track the same length, therefore nothing should change
        scale = 1.0
        expected = 90, 200
        actual = DUMMY_DATASET.scale_slice_indices(slice_start, slice_end, scale)
        self.assertEqual(expected, actual)

    def test_adding_condition_tokens_does_not_change_score(self):
        """Input IDs with + without condition tokens should decode to the same score"""
        token_factory = prepare_conditioned_tokenizer()
        kwargs = dict(
            tokenizer=token_factory,
            files_paths=[TEST_MIDI1, TEST_MIDI2, TEST_MIDI3],
            max_seq_len=512,
            do_conditioning=True,
            do_augmentation=False
        )
        # This test will work with multiple dataset types
        for ds_cls in [DatasetMIDIConditioned, DatasetMIDIConditionedRandomChunk]:
            ds = ds_cls(**kwargs)
            # Iterate over every item
            for i in ds:
                # Unpack the input IDs (with conditioning) and condition tokens
                input_ids, condition_tokens = i["input_ids"], i["condition_ids"]
                # Remove padding tokens from the condition IDs
                condition_tokens = torch.tensor([i for i in condition_tokens if i != token_factory.pad_token_id])
                # Remove the condition tokens from the input ids
                raw_input_ids = input_ids[len(condition_tokens):]
                self.assertFalse(set(condition_tokens.tolist()) & set(raw_input_ids.tolist()))  # should be no overlap
                # Serialise tokens into score
                # First, with condition tokens
                score_with_condition = token_factory.decode(input_ids.unsqueeze(0))
                # Second, without condition tokens
                score_without_condition = token_factory.decode(raw_input_ids.unsqueeze(0))
                # Scores should be equivalent whether they have condition tokens or not
                #  i.e., condition tokens should decode into nothing "musical" on the MIDI side of things
                self.assertTrue(scores_are_identical(score_with_condition, score_without_condition))

    def test_getitem(self):
        # Iterate over all slices of all tracks
        for item in DUMMY_DATASET:
            input_ids, targets = item["input_ids"].tolist(), item["labels"].tolist()
            # Should be the desired length
            self.assertEqual(len(input_ids), 512)
            self.assertEqual(len(targets), 512)
            self.assertEqual(input_ids[1:], targets[:-1])
            # We're not conditioning here, so we shouldn't have any conditioning tokens
            for tok in input_ids:
                for t in CONDITION_TOKEN_STARTS:
                    self.assertFalse(TOKENIZER[tok].startswith(t))
        # Now, just try the first slice
        slice1 = DUMMY_DATASET.__getitem__(0)
        slice1_input_ids = slice1["input_ids"].tolist()
        # First token should be BOS
        self.assertTrue(slice1_input_ids[0] == TOKENIZER["BOS_None"])
        # Should not have any padding tokens here
        self.assertFalse(TOKENIZER["PAD_None"] in slice1_input_ids)

        # Finally, try the last slice of the last item
        slice_last = DUMMY_DATASET.__getitem__(-1)
        slice_last_input_ids = slice_last["input_ids"].tolist()
        # Should have the EOS token at some point here
        no_pad = [i_ for i_ in slice_last_input_ids if i_ != TOKENIZER["PAD_None"]]
        self.assertTrue(no_pad[-1] == TOKENIZER["EOS_None"])
        # We should have some padding tokens here
        self.assertTrue(TOKENIZER["PAD_None"] in slice_last_input_ids)

    def test_getitem_with_augmentation(self):
        kwargs = dict(
            tokenizer=TOKENIZER,
            files_paths=[TEST_MIDI1],  # this track has a bpm of 297-ish and a time signature of 4/4
            do_augmentation=True,
            do_conditioning=False,
            max_seq_len=100,
        )
        for ds_cls in [DatasetMIDIConditionedRandomChunk, DatasetMIDIConditioned]:
            ds = ds_cls(**kwargs)
            # Test the first "slice" of the first item
            item = ds.__getitem__(0)
            input_ids, targets = item["input_ids"].tolist(), item["labels"].tolist()
            # Should be the desired length
            self.assertEqual(len(input_ids), 100)
            self.assertEqual(len(targets), 100)

    def test_getitem_with_conditioning_midi1(self):
        token_factory = prepare_conditioned_tokenizer()
        # Create the dataset with MIDI file 1
        kwargs = dict(
            tokenizer=token_factory,
            files_paths=[TEST_MIDI1],  # this track has a bpm of 297-ish and a time signature of 4/4
            do_augmentation=False,
            max_seq_len=100,
            do_conditioning=True,
        )
        # We can run this test with multiple dataset types
        for ds_cls in [DatasetMIDIConditionedRandomChunk, DatasetMIDIConditioned]:
            ds = ds_cls(**kwargs)
            # Test the first "slice" of the first item
            item = ds.__getitem__(0)
            input_ids, targets = item["input_ids"].tolist(), item["labels"].tolist()
            # Should be the desired length
            self.assertEqual(len(input_ids), 100)
            self.assertEqual(len(targets), 100)
            # Input IDs should start with condition tokens, followed by BOS
            #  IDs are sorted in order of GENRE -> PIANIST -> TEMPO -> TIMESIG -> BOS
            #  GENRE and PIANIST are sorted in DESCENDING weight order, with the track pianist always placed first
            self.assertEqual(input_ids[0], token_factory["GENRES_Caribbean"])  # most strongly weighted genre, = 10
            self.assertEqual(input_ids[1], token_factory["GENRES_HardBop"])
            self.assertEqual(input_ids[2], token_factory["GENRES_PostBop"])
            self.assertEqual(input_ids[3], token_factory["GENRES_StraightAheadJazz"])
            self.assertEqual(input_ids[4], token_factory["GENRES_Fusion"])  # least strongly weighted genre, = 5
            self.assertEqual(input_ids[5], token_factory["PIANIST_KennyBarron"])
            self.assertEqual(input_ids[6], token_factory["TEMPOCUSTOM_300"])  # closest match to our provided tempo
            self.assertEqual(input_ids[7], token_factory["TIMESIGNATURECUSTOM_44"])

        # These tests can only work with our exhaustive dataloader
        ds = DatasetMIDIConditioned(**kwargs)
        # Test the first slice of the first item
        item = ds.__getitem__(0)
        input_ids, targets = item["input_ids"].tolist(), item["labels"].tolist()
        self.assertEqual(input_ids[8], token_factory["BOS_None"])  # after condition tokens, we should get BOS

        # Test the last "slice" of the first item
        item = ds.__getitem__(-1)
        input_ids, targets = item["input_ids"].tolist(), item["labels"].tolist()
        # Should be the desired length
        self.assertEqual(len(input_ids), 100)
        self.assertEqual(len(targets), 100)
        # Should have all the desired condition tokens, but not followed by BOS
        self.assertEqual(input_ids[0], token_factory["GENRES_Caribbean"])  # most strongly weighted genre, = 10
        self.assertEqual(input_ids[7], token_factory["TIMESIGNATURECUSTOM_44"])
        self.assertNotEqual(input_ids[8], token_factory["BOS_None"])  # after condition tokens, NO BOS
        # Should have the EOS token in the last chunk
        no_pad = [i_ for i_ in input_ids if i_ != token_factory["PAD_None"]]
        self.assertTrue(no_pad[-1] == token_factory["EOS_None"])

    def test_getitem_with_conditioning_midi2(self):
        token_factory = prepare_conditioned_tokenizer()
        # Create the dataset with MIDI file 2
        kwargs = dict(
            tokenizer=token_factory,
            files_paths=[TEST_MIDI2],
            do_augmentation=False,
            max_seq_len=100,
            do_conditioning=True,
        )
        # We can run this test with multiple dataset types
        for ds_cls in [DatasetMIDIConditionedRandomChunk, DatasetMIDIConditioned]:
            # Create the dataset with MIDI file 2
            ds = ds_cls(**kwargs)

            # Get the first slice
            item = ds.__getitem__(0)
            input_ids, targets = item["input_ids"].tolist(), item["labels"].tolist()
            # Should be the desired length
            self.assertEqual(len(input_ids), 100)
            self.assertEqual(len(targets), 100)
            # This track has one GENRE token, associated with the pianist
            #  It also has two SIMILAR PIANISTS
            self.assertEqual(input_ids[0], token_factory["GENRES_StraightAheadJazz"])
            self.assertEqual(input_ids[1], token_factory["PIANIST_BradMehldau"])  # similar to Beegie Adair
            self.assertEqual(input_ids[2], token_factory["PIANIST_KennyDrew"])
            self.assertEqual(targets[0], token_factory["PIANIST_BradMehldau"])
            self.assertEqual(targets[1], token_factory["PIANIST_KennyDrew"])
            # We should not have any tempo or time signature tokens for this track
            for tok in input_ids:
                for t in ["TEMPO", "TIMESIGNATURE"]:
                    self.assertFalse(token_factory[tok].startswith(t))

        # These tests can only work with our exhaustive dataloader
        ds = DatasetMIDIConditioned(**kwargs)
        # Test the first slice of the first item
        item = ds.__getitem__(0)
        input_ids, targets = item["input_ids"].tolist(), item["labels"].tolist()
        self.assertEqual(input_ids[3], token_factory["BOS_None"])
        # Get the last slice
        item = ds.__getitem__(-1)
        input_ids, targets = item["input_ids"].tolist(), item["labels"].tolist()
        # Should be the desired length
        self.assertEqual(len(input_ids), 100)
        self.assertEqual(len(targets), 100)
        # Test conditioning tokens in correct place
        self.assertEqual(input_ids[0], token_factory["GENRES_StraightAheadJazz"])
        self.assertEqual(input_ids[2], token_factory["PIANIST_KennyDrew"])
        self.assertNotEqual(input_ids[3], token_factory["BOS_None"])  # no BOS for chunks other than first
        # Should have the EOS token in the last chunk
        no_pad = [i_ for i_ in input_ids if i_ != token_factory["PAD_None"]]
        self.assertTrue(no_pad[-1] == token_factory["EOS_None"])

    def test_getitem_with_conditioning_midi3(self):
        token_factory = prepare_conditioned_tokenizer()
        kwargs = dict(
            tokenizer=token_factory,
            files_paths=[TEST_MIDI3],
            do_augmentation=False,
            max_seq_len=100,
            do_conditioning=True,
        )
        # We can run this test with multiple dataset types
        for ds_cls in [DatasetMIDIConditionedRandomChunk, DatasetMIDIConditioned]:
            # Create the dataset with MIDI file 3
            ds = ds_cls(**kwargs)
            # Test the first slice
            item = ds.__getitem__(0)
            input_ids, targets = item["input_ids"].tolist(), item["labels"].tolist()
            # Input IDs should NOT start with any conditioning tokens
            for tok in input_ids:
                for t in CONDITION_TOKEN_STARTS:
                    self.assertFalse(token_factory[tok].startswith(t))
            # Should be the desired length
            self.assertEqual(len(input_ids), 100)
            self.assertEqual(len(targets), 100)

    @unittest.skipIf(os.getenv("REMOTE") == "true", "Skipping test on GitHub Actions")
    def test_getitem_full_dataset(self, to_test: int = 100):
        """Test our dataset with a large number of tracks (will be skipped on remote)"""

        def runner(ds):
            # Get the list of condition tokens from our tokenizer
            all_condition_tokens = [i for i in ds.tokenizer.vocab.keys() if i.startswith(tuple(CONDITION_TOKEN_STARTS))]
            # Iterate over every track in the dataset
            for item in tqdm(ds, desc=f"Checking dataset with {to_test} tracks"):
                # Get the input IDs, targets, and condition_ids
                input_ids, targets, condition_ids = item["input_ids"], item["labels"], item["condition_ids"]

                # Remove padding from the condition IDs
                condition_ids = torch.tensor([i for i in condition_ids if i != ds.tokenizer.pad_token_id])
                self.assertTrue(input_ids.tolist()[1:] == targets.tolist()[:-1])

                # Remove the condition tokens from the input ids
                raw_input_ids = input_ids[len(condition_ids):]
                self.assertFalse(set(condition_ids.tolist()) & set(raw_input_ids.tolist()))  # no overlap

                # Serialise tokens into score: should be equivalent
                score_with_condition = ds.tokenizer.decode(input_ids.unsqueeze(0))
                score_without_condition = ds.tokenizer.decode(raw_input_ids.unsqueeze(0))
                self.assertTrue(scores_are_identical(score_with_condition, score_without_condition))

                # Iterate over "track" tokens (input IDs, targets)
                decoded = decoder(ds.tokenizer, raw_input_ids.tolist())
                for decoded_token in decoded:
                    # When decoded, input tokens should not be included in our list of condition tokens
                    self.assertFalse(decoded_token in all_condition_tokens)
                    # When decoded, input tokens should NOT start with condition token patterns
                    self.assertFalse(decoded_token.startswith(tuple(CONDITION_TOKEN_STARTS)))

                # Test that there are no more than 5 pianist and genre tokens for one recording
                decoded_condition_tokens = decoder(ds.tokenizer, condition_ids.tolist())
                for tok_start in ["PIANIST", "GENRE"]:
                    ts = [i for i in decoded_condition_tokens if i.startswith(tok_start)]
                    self.assertTrue(len(ts) <= 5)
                # Iterate over "condition" tokens
                for decoded_token in decoded_condition_tokens:
                    # When decoded, condition tokens SHOULD be in our list of all condition tokens
                    self.assertTrue(decoded_token in all_condition_tokens)
                    # When decoded, condition tokens SHOULD start with one of our condition token patterns
                    self.assertTrue(decoded_token.startswith(tuple(CONDITION_TOKEN_STARTS)))

        # Get a large number of tracks
        midi_fps = utils.get_data_files_with_ext("data/raw", "**/*.mid")
        random.shuffle(midi_fps)
        midi_fps = midi_fps[:to_test]
        # Get the arguments for the dataset
        kwargs = dict(
            files_paths=midi_fps,
            max_seq_len=utils.MAX_SEQUENCE_LENGTH,
            do_augmentation=False,
            do_conditioning=True,
            n_clips=to_test
        )
        # This test works with both tokenizer classes
        for ds_cls in [DatasetMIDIConditionedRandomChunk, DatasetMIDIConditioned]:
            # Create a tokenizer
            tok = load_tokenizer(tokenizer_str="midilike", )
            add_tempos_to_vocab(tok, (80, 300), 32)
            add_timesignatures_to_vocab(tok, [3, 4])
            add_pianists_to_vocab(tok)
            add_genres_to_vocab(tok)
            # FIRST: we test without training the tokenizer
            dataset = ds_cls(tokenizer=tok, **kwargs)
            runner(dataset)
            # SECOND: we test WITH training the tokenizer
            train_tokenizer(tok, midi_fps, vocab_size=1000, training_method="BPE")
            dataset = ds_cls(tokenizer=tok, **kwargs)
            runner(dataset)

    def test_getitem_consistency_across_epochs(self):
        """Test that we don't modify the underlying preloaded objects across successive epochs with augmentation"""
        # Create the dataset
        tok = prepare_conditioned_tokenizer()
        kwargs = dict(
            tokenizer=tok,
            files_paths=[TEST_MIDI1, TEST_MIDI2, TEST_MIDI3],
            max_seq_len=utils.MAX_SEQUENCE_LENGTH,
            do_augmentation=True,  # need augmentation for this to work properly
            do_conditioning=True
        )
        # We can run this test with multiple dataset types
        for ds_cls in [DatasetMIDIConditionedRandomChunk, DatasetMIDIConditioned]:
            ds = ds_cls(**kwargs)
            before_augment = deepcopy(ds.preloaded_data[0])
            # Create the item a few times: this will apply augmentation to the item in .track_slices[0]
            for _ in range(20):
                _ = ds.__getitem__(0)
            # Check that we haven't manipulated the underlying item at all
            after_augment = ds.preloaded_data[0]
            self.assertEqual(before_augment[-1]["tempo"], after_augment[-1]["tempo"])  # check tempo field in metadata
            self.assertTrue(scores_are_identical(before_augment[0], after_augment[0]))  # check score items
            self.assertEqual(before_augment[1], after_augment[1])  # check slices

    def test_chunk_starting_point(self):
        # Create the dataset
        tok = prepare_conditioned_tokenizer()
        train_tokenizer(tok, [TEST_MIDI1, TEST_MIDI2, TEST_MIDI3])
        # This test is only relevant for our random chunk dataloader
        ds = DatasetMIDIConditionedRandomChunk(
            tokenizer=tok,
            files_paths=[TEST_MIDI1, TEST_MIDI2, TEST_MIDI3],
            max_seq_len=utils.MAX_SEQUENCE_LENGTH,
            do_augmentation=False,
            do_conditioning=False
        )
        tokseq = ds.preloaded_data[0][0]
        tokseq_ids = ds.score_to_token_sequence(tokseq)
        for _ in range(50):
            starting_point = ds.get_slice_start_point(tokseq_ids)
            chunked = tokseq_ids[starting_point]
            detokenized = ds.tokenizer[ds.tokenizer.bpe_token_mapping[chunked][0]]
            # The token should be one of our valid token types
            self.assertTrue(detokenized.startswith(ds.START_TOKENS))
            # Chunking the sequence with this starting point should lead to sequences longer than our desired length
            self.assertTrue(len(tokseq_ids[chunked:chunked + ds.max_seq_len]) >= ds.min_seq_len)


if __name__ == '__main__':
    utils.seed_everything(utils.SEED)
    unittest.main()
