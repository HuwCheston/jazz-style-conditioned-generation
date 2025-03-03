#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for dataloader"""

import os
import unittest

from miditok import REMI
from symusic import Score, Note

from jazz_style_conditioned_generation import utils
from jazz_style_conditioned_generation.data.dataloader import *

TEST_RESOURCES = os.path.join(utils.get_project_root(), "tests/test_resources")
TEST_MIDI = os.path.join(TEST_RESOURCES, "test_midi1/piano_midi.mid")


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
            max_seq_len=512,
            do_conditioning=False
        )
        gotitem = ds.__getitem__(0)
        input_ids, targets = gotitem["input_ids"], gotitem["labels"]
        # Input IDs and labels should be the target length
        self.assertEqual(input_ids.size(0), 512)
        self.assertEqual(targets.size(0), 512)
        # Labels should be the input IDs shifted by one
        self.assertEqual(input_ids.tolist()[1:], targets.tolist()[:-1])
        # We shouldn't have any 0s (i.e., pad tokens) in our attention mask
        self.assertFalse(True in gotitem["attention_mask"])

    def test_dataset_random_chunk(self):
        max_seq_length = 10
        ds = DatasetMIDIRandomChunk(
            tokenizer=REMI(),
            files_paths=[TEST_MIDI],
            do_augmentation=False,
            max_seq_len=10,
            do_conditioning=False
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
        token_factory = REMI()
        token_factory.add_to_vocab("PIANIST_KennyBarron")
        token_factory.add_to_vocab("GENRES_HardBop")
        ds = DatasetMIDIRandomChunk(
            tokenizer=token_factory,
            files_paths=[TEST_MIDI],
            do_augmentation=False,
            max_seq_len=10,
            do_conditioning=True,
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
        tokenizer = REMI()
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
            tokenizer=REMI(),
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
        token_factory = REMI()
        token_factory.add_to_vocab("PIANIST_KennyBarron")
        token_factory.add_to_vocab("GENRES_HardBop")
        ds = DatasetMIDIExhaustive(
            tokenizer=token_factory,
            files_paths=[TEST_MIDI],
            do_augmentation=False,
            max_seq_len=10,
            do_conditioning=True,
            condition_mapping={
                "pianist": {"Kenny Barron": "PIANIST_KennyBarron"},
                "genres": {"Hard Bop": "GENRES_HardBop"},
            }
        )
        item = ds.__getitem__(0)
        input_ids, targets = item["input_ids"].tolist(), item["labels"].tolist()
        # Input IDs should start with BOS, followed by the expected conditioning tokens
        self.assertEqual(input_ids[0], token_factory["BOS_None"])
        self.assertEqual(input_ids[1], token_factory["GENRES_HardBop"])
        self.assertEqual(input_ids[2], token_factory["PIANIST_KennyBarron"])
        self.assertEqual(targets[0], token_factory["GENRES_HardBop"])
        self.assertEqual(targets[1], token_factory["PIANIST_KennyBarron"])
        # Should be the desired length
        self.assertEqual(len(input_ids), 10)
        self.assertEqual(len(targets), 10)
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
        token_factory = REMI()
        # This track has the tag "Calypso". We are expecting this to be merged with the tag "GENRES_Caribbean"
        token_factory.add_to_vocab("PIANIST_KennyBarron")
        token_factory.add_to_vocab("GENRES_HardBop")
        token_factory.add_to_vocab("GENRES_Caribbean")
        ds = DatasetMIDIExhaustive(
            tokenizer=token_factory,
            files_paths=[TEST_MIDI],
            do_augmentation=False,
            max_seq_len=10,
            do_conditioning=True,
            condition_mapping={
                "pianist": {"Kenny Barron": "PIANIST_KennyBarron"},
                "genres": {"Caribbean": "GENRES_Caribbean", "Hard Bop": "GENRES_HardBop"},
            }
        )
        item = ds.__getitem__(0)
        input_ids, targets = item["input_ids"].tolist(), item["labels"].tolist()
        # Input IDs should start with BOS, followed by the expected conditioning tokens
        self.assertEqual(input_ids[0], token_factory["BOS_None"])
        self.assertEqual(input_ids[1], token_factory["GENRES_Caribbean"])
        self.assertEqual(input_ids[2], token_factory["GENRES_HardBop"])
        self.assertEqual(input_ids[3], token_factory["PIANIST_KennyBarron"])
        self.assertEqual(targets[0], token_factory["GENRES_Caribbean"])
        self.assertEqual(targets[1], token_factory["GENRES_HardBop"])
        self.assertEqual(targets[2], token_factory["PIANIST_KennyBarron"])
        # Should be the desired length
        self.assertEqual(len(input_ids), 10)
        self.assertEqual(len(targets), 10)
        # Testing the final chunk
        final_item = ds.__getitem__(len(ds) - 1)
        input_ids, targets = final_item["input_ids"].tolist(), final_item["labels"].tolist()
        # Should start with the condition tokens now
        self.assertEqual(input_ids[0], token_factory["GENRES_Caribbean"])
        self.assertEqual(input_ids[1], token_factory["GENRES_HardBop"])
        self.assertEqual(input_ids[2], token_factory["PIANIST_KennyBarron"])
        self.assertEqual(targets[0], token_factory["GENRES_HardBop"])
        self.assertEqual(targets[1], token_factory["PIANIST_KennyBarron"])
        # and end with the EOS token, after padding is removed
        input_ids_no_pad = [i for i in input_ids if i != token_factory["PAD_None"]]
        self.assertEqual(input_ids_no_pad[-1], token_factory["EOS_None"])

    def test_non_existing_filepath(self):
        tokenizer = REMI()
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

    def test_remove_short_notes_dummy(self):
        # Test with a dummy example
        notelist = [
            # Keep this one
            Note(pitch=80, duration=100, time=100, velocity=80, ttype="tick"),
            # Remove this one
            Note(pitch=70, duration=1, time=100, velocity=80, ttype="tick")
        ]
        expected = [Note(pitch=80, duration=100, time=100, velocity=80, ttype="tick")]
        actual = remove_short_notes(notelist)
        self.assertEqual(actual, expected)

    def test_remove_short_notes_real(self):
        # Test with a real example
        # The example has 4 notes at pitch 68 (G#4): three are of a normal length, one is really short
        real_example = Score(os.path.join(TEST_RESOURCES, "test_midi_repeatnotes.mid"))
        # Test the input
        notelist = real_example.tracks[0].notes
        init_len = len([i for i in notelist if i.pitch == 68])
        self.assertEqual(init_len, 4)
        # Test the output
        actual = remove_short_notes(notelist)
        actual_len = len([i for i in actual if i.pitch == 68])
        self.assertEqual(actual_len, 3)
        self.assertLess(actual_len, init_len)

    def test_merge_repeated_notes_dummy(self):
        # These notes should be merged into one
        notelist = [
            Note(pitch=80, duration=5, time=100, velocity=80, ttype="tick"),
            Note(pitch=80, duration=5, time=110, velocity=90, ttype="tick"),  # velocities will be merged too
        ]
        # Duration should be note1_duration + (note2_onset - note1_onset) + note2_duration
        expected = [Note(pitch=80, duration=15, time=100, velocity=85, ttype="tick")]
        actual = merge_repeated_notes(notelist)
        self.assertEqual(actual, expected)
        # Even these notes are adjacent, they shouldn't be merged as they are different pitches
        notelist = [
            Note(pitch=90, duration=5, time=110, velocity=80, ttype="tick"),
            Note(pitch=91, duration=5, time=120, velocity=80, ttype="tick"),
        ]
        expected = notelist
        actual = merge_repeated_notes(notelist)
        self.assertEqual(actual, expected)
        # These notes are not adjacent, so shouldn't be touched
        notelist = [

            Note(pitch=90, duration=5, time=110, velocity=80, ttype="tick"),
            Note(pitch=90, duration=5, time=1000, velocity=80, ttype="tick"),
        ]
        expected = notelist
        actual = merge_repeated_notes(notelist)
        self.assertEqual(actual, expected)

    def test_merge_repeated_notes_real(self):
        # Test with a real example
        # The example has 8 notes at pitch 75 (D#5): two of these should be merged together to make 7 total notes
        real_example = Score(os.path.join(TEST_RESOURCES, "test_midi_repeatnotes.mid"))
        # Test before processing
        notelist = real_example.tracks[0].notes
        init_len = len([i for i in notelist if i.pitch == 75])
        self.assertEqual(8, init_len)
        # Test after processing
        actual = merge_repeated_notes(notelist)
        actual_len = len([i for i in actual if i.pitch == 75])
        self.assertEqual(7, actual_len)
        self.assertLess(actual_len, init_len)

    def test_notelist_to_score(self):
        notelist = [Note(pitch=80, duration=100, time=100, velocity=80, ttype="tick")]
        actual = note_list_to_score(notelist, 100)
        self.assertTrue(isinstance(actual, Score))
        self.assertTrue(len(actual.tracks) == 1)
        self.assertTrue(len(actual.tracks[0].notes) == 1)
        self.assertTrue(actual.tracks[0].notes[0].time == 100)
        self.assertTrue(actual.tracks[0].notes[0].duration == 100)
        self.assertTrue(actual.tracks[0].notes[0].pitch == 80)

    def test_add_condition_tokens_to_sequence(self):
        dummy = [1, 3, 3, 3, 5, 6]
        condition_tokens = [100, 200, 300]
        # Treat 1 as BOS token: condition tokens should be added after
        expected_inputs = [1, 100, 200, 300, 3, 3]
        expected_targets = [100, 200, 300, 3, 3, 3]
        actual_inputs, actual_targets = add_condition_tokens_to_sequence(dummy, condition_tokens, bos_token_id=1)
        self.assertEqual(expected_inputs, actual_inputs)
        self.assertEqual(expected_targets, actual_targets)
        # Treat 1000 as BOS token: condition tokens should be added to start of sequence
        expected_inputs = [100, 200, 300, 1, 3, 3]
        expected_targets = [200, 300, 1, 3, 3, 3]
        actual_inputs, actual_targets = add_condition_tokens_to_sequence(dummy, condition_tokens, bos_token_id=1000)
        self.assertEqual(expected_inputs, actual_inputs)
        self.assertEqual(expected_targets, actual_targets)
        # Testing with adding no condition tokens
        condition_tokens = []
        self.assertRaises(AssertionError, add_condition_tokens_to_sequence, dummy, condition_tokens, 1)

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

    def test_dataset_attention_mask(self):
        # Test exhaustive dataloader
        token_factory = REMI()
        ds = DatasetMIDIExhaustive(
            tokenizer=token_factory,
            files_paths=[TEST_MIDI],
            do_augmentation=False,
            max_seq_len=10,
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
            do_conditioning=False
        )
        # Random chunks FROM THIS TRACK (WHICH IS LONG) should not have any padding
        #  NB., if we had a track which was very short, we would expect some padding here
        first_chunk = ds.__getitem__(0)
        self.assertFalse(True in first_chunk["attention_mask"])


if __name__ == '__main__':
    unittest.main()
