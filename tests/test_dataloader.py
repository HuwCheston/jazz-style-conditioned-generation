#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for dataloader"""

import os
import random
import unittest

from miditok import REMI, MIDILike, TokenizerConfig
from pretty_midi import Note as PMNote
from pretty_midi import PrettyMIDI, Instrument
from pretty_midi import TimeSignature as PMTimeSignature
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
        # Test a few times
        for _ in range(10):
            returned = randomly_slice_sequence(tokseq, desired_len, end_overlap=0.)
            self.assertTrue(len(returned) < len(tokseq))
            self.assertEqual(len(returned), desired_len)

    def test_dataset_random_chunk_getitem(self):
        ds = DatasetMIDIRandomChunk(
            tokenizer=REMI(),
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
            tokenizer=REMI(),
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
        token_factory = REMI()
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
        token_factory = REMI()
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

    def test_dataset_attention_mask(self):
        # Test exhaustive dataloader
        token_factory = REMI()
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
        token_factory = REMI()
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


class PreProcessingScoreTest(unittest.TestCase):
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
        real_example = load_score(os.path.join(TEST_RESOURCES, "test_midi_repeatnotes.mid"))
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
            Note(pitch=80, duration=5, time=105, velocity=90, ttype="tick"),  # velocities will be merged too
        ]
        # Duration should be note1_duration + (note2_onset - note1_onset) + note2_duration
        expected = [Note(pitch=80, duration=10, time=100, velocity=85, ttype="tick")]
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
        real_example = load_score(os.path.join(TEST_RESOURCES, "test_midi_repeatnotes.mid"))
        # Test before processing
        notelist = real_example.tracks[0].notes
        init_len = len([i for i in notelist if i.pitch == 75])
        self.assertEqual(8, init_len)
        # Test after processing
        actual = merge_repeated_notes(notelist, overlap_ticks=25)
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
        # Condition tokens should be added before the start of the sequence, and it should be truncated to fit
        expected_inputs = [100, 200, 300, 1, 3, 3]
        expected_targets = [200, 300, 1, 3, 3, 3]
        actual_inputs, actual_targets = add_condition_tokens_to_sequence(dummy, condition_tokens)
        self.assertEqual(expected_inputs, actual_inputs)
        self.assertEqual(expected_targets, actual_targets)
        # Testing with adding no condition tokens
        condition_tokens = []
        self.assertRaises(AssertionError, add_condition_tokens_to_sequence, dummy, condition_tokens)

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

    def test_remove_invalid_notes(self):
        notes = [
            # This note is valid
            Note(pitch=90, duration=5, time=110, velocity=80, ttype="tick"),
            # This note is too low
            Note(pitch=1, duration=10, time=1000, velocity=50, ttype="tick"),
            # This note is too high
            Note(pitch=125, duration=10, time=2000, velocity=50, ttype="tick"),
            # This note is valid
            Note(pitch=50, duration=10, time=3000, velocity=60, ttype="tick"),
        ]
        expected_len = 2
        actual_len = len(remove_out_of_range_notes(notes))
        self.assertEqual(expected_len, actual_len)


class LoadScoreTest(unittest.TestCase):
    def test_load_score_dummy(self):
        # Create the tokenizer
        TOKENIZER = MIDILike(
            TokenizerConfig(
                # This means that we will have exactly 100 evenly-spaced tokens per "bar"
                beat_res={(0, utils.TIME_SIGNATURE): 100 // utils.TIME_SIGNATURE}
            )
        )
        # The tokenizer vocab is a dictionary: we just want to get the TimeShift tokens
        timeshifts = [m for m in TOKENIZER.vocab.keys() if "TimeShift" in m]
        n_timeshifts = len(timeshifts)  # 100 tokens at evenly-spaced 10ms increments between 0 - 1 second
        assert n_timeshifts == 100
        # Get tokens corresponding to particular time values
        token_10ms = timeshifts[0]
        token_50ms = timeshifts[4]
        token_100ms = timeshifts[9]
        token_500ms = timeshifts[49]
        token_1000ms = timeshifts[-1]
        # Create some dummy pretty MIDI notes with standard durations
        dummy_notes = [
            PMNote(start=0., end=1., pitch=80, velocity=50),  # 1 second duration
            PMNote(start=1., end=1.1, pitch=81, velocity=50),  # 100 millisecond duration
            PMNote(start=1.1, end=1.11, pitch=82, velocity=50),  # 10 millisecond duration
            PMNote(start=1.11, end=1.16, pitch=83, velocity=50),  # 50 millisecond duration
            PMNote(start=1.16, end=1.66, pitch=84, velocity=50),  # 500 millisecond duration
            PMNote(start=1.66, end=3.66, pitch=85, velocity=50),  # 2 second duration
            PMNote(start=3.66, end=5.67, pitch=86, velocity=50),  # 2 second + 10 millisecond duration
        ]
        # Convert our dummy note stream into the expected tokens
        expected_tokens = [
            'NoteOn_80', 'Velocity_51', token_1000ms, 'NoteOff_80',
            'NoteOn_81', 'Velocity_51', token_100ms, 'NoteOff_81',
            'NoteOn_82', 'Velocity_51', token_10ms, 'NoteOff_82',
            'NoteOn_83', 'Velocity_51', token_50ms, 'NoteOff_83',
            'NoteOn_84', 'Velocity_51', token_500ms, 'NoteOff_84',
            'NoteOn_85', 'Velocity_51', token_1000ms, token_1000ms, 'NoteOff_85',
            'NoteOn_86', 'Velocity_51', token_1000ms, token_1000ms, token_10ms, 'NoteOff_86',
        ]
        # Create a fake MIDI file
        # Get a random tempo and time signature
        tempo = random.randint(100, 300)
        ts = random.randint(1, 8)
        tpq = random.randint(100, 1000)
        # Create a PrettyMIDI object at the desired resolution with a random tempo
        pm = PrettyMIDI(resolution=tpq, initial_tempo=tempo)
        pm.instruments = [Instrument(program=0)]
        # Add some notes in
        pm.instruments[0].notes = dummy_notes
        # Add some random time signature and tempo changes
        pm.time_signature_changes = [PMTimeSignature(ts, 4, 0)]
        # Sanity check the input
        self.assertEqual(pm.resolution, tpq)
        self.assertEqual(pm.time_signature_changes[0].numerator, ts)
        self.assertEqual(round(pm.get_tempo_changes()[1][0]), tempo)
        # Write our random midi file
        out_path = "temp_pm.mid"
        pm.write(out_path)
        # Load in as a symusic score object and apply our preprocessing to standardise tempo, time signature, & TPQ
        score = load_score(out_path)
        # Sanity check that the score is correct
        self.assertTrue(score.ticks_per_quarter == utils.TICKS_PER_QUARTER)
        self.assertTrue(len(score.tempos) == 1)
        self.assertTrue(score.tempos[0].qpm == utils.TEMPO)
        self.assertTrue(len(score.time_signatures) == 1)
        self.assertTrue(score.time_signatures[0].numerator == utils.TIME_SIGNATURE)
        # Tokenize the output
        toks = TOKENIZER.encode(score)[0].tokens
        # Sanity check that the tokens are correct
        only_timeshifts = [t for t in toks if "TimeShift" in t]
        self.assertTrue(only_timeshifts[0] == token_1000ms)
        self.assertTrue(only_timeshifts[1] == token_100ms)
        self.assertTrue(only_timeshifts[2] == token_10ms)
        self.assertTrue(only_timeshifts[3] == token_50ms)
        self.assertTrue(only_timeshifts[4] == token_500ms)
        self.assertTrue(toks == expected_tokens)
        # Clean up
        os.remove(out_path)

    def test_load_score_real(self):
        # TESTING WITH ACTUAL MIDI FILES
        files = [
            "test_midi1/piano_midi.mid",
            "test_midi2/piano_midi.mid",
            "test_midi3/piano_midi.mid",
            "test_midi_jja1/piano_midi.mid",
            "test_midi_bushgrafts1/piano_midi.mid",
            "test_midi_bushgrafts2/piano_midi.mid",
            "test_midi_bushgrafts3/piano_midi.mid",
        ]
        for file in files:
            file = os.path.join(utils.get_project_root(), "tests/test_resources", file)
            # Load with Symusic and ttype=="Second" to get actual times
            pm_load = Score(file, ttype="Second")
            # Load with our custom symusic function
            sm_load = load_score(file)
            # Get the notes from both files and sort by onset time (not sure if this is necessary)
            pm_notes = sorted(pm_load.tracks[0].notes, key=lambda x: x.start)
            sm_notes = sorted(sm_load.tracks[0].notes, key=lambda x: x.start)
            # Number of notes should be the same
            self.assertTrue(len(pm_notes) == len(sm_notes))
            # Start times for notes should be directly equivalent
            for pm, sm in zip(pm_notes, sm_notes):
                self.assertTrue(utils.base_round(pm.start * 1000, 10) == sm.time)  # make sure to round!
                self.assertTrue(utils.base_round(pm.duration * 1000, 10) == sm.duration)


if __name__ == '__main__':
    utils.seed_everything(utils.SEED)
    unittest.main()
