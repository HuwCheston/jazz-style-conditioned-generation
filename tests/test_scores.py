#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for score preprocessing"""

import os
import random
import unittest

from miditok import MIDILike, TokenizerConfig, TSD
from pretty_midi import Note as PMNote
from pretty_midi import PrettyMIDI, Instrument
from pretty_midi import TimeSignature as PMTimeSignature
from symusic import Score, Note
from symusic.core import Second
from tqdm import tqdm

from jazz_style_conditioned_generation import utils
from jazz_style_conditioned_generation.data.scores import (
    remove_short_notes,
    load_score,
    merge_repeated_notes,
    note_list_to_score,
    remove_out_of_range_notes,
    get_notes_from_score,
    preprocess_score,
    cap_long_notes,
    remove_overlap,
    remove_duplicate_notes,
    align_to_start
)

TEST_RESOURCES = os.path.join(utils.get_project_root(), "tests/test_resources")


class PreProcessingScoreTest(unittest.TestCase):
    def test_preprocess_as_seconds(self):
        # Test with a dummy example
        notelist = [
            # Keep this one
            Note(pitch=80, duration=1.0, time=100, velocity=80, ttype="Second"),
            # Remove this one
            Note(pitch=70, duration=0.001, time=100, velocity=80, ttype="Second"),
            # remove this one
            Note(pitch=60, duration=0.004, time=100, velocity=80, ttype="Second")
        ]
        # Convert the notelist to a score with time in seconds
        score = note_list_to_score(notelist, utils.TICKS_PER_QUARTER, ttype="Second")
        self.assertTrue(isinstance(score.ttype, Second))
        # Preprocess the score
        preproc = preprocess_score(score)
        # Should have removed all but the very first note
        self.assertTrue(len(preproc.tracks[0].notes) == 1)
        self.assertTrue(preproc.tracks[0].notes[0].pitch == 80)
        self.assertTrue(isinstance(preproc.ttype, Second))

    def test_align_notes_to_start(self):
        # Test with a dummy example, using ticks
        notelist = [
            Note(pitch=80, duration=100, time=100, velocity=80, ttype="tick"),
            Note(pitch=70, duration=10, time=120, velocity=80, ttype="tick")
        ]
        expected = [
            Note(pitch=80, duration=100, time=0, velocity=80, ttype="tick"),
            Note(pitch=70, duration=10, time=20, velocity=80, ttype="tick")
        ]
        actual = align_to_start(notelist)
        self.assertTrue(expected == actual)
        # Test with a dummy example, using seconds
        notelist = [
            Note(pitch=80, duration=0.5, time=0.15, velocity=80, ttype="Second"),
            Note(pitch=70, duration=0.6, time=0.35, velocity=80, ttype="Second")
        ]
        expected = [
            Note(pitch=80, duration=0.5, time=0., velocity=80, ttype="Second"),
            Note(pitch=70, duration=0.6, time=0.2, velocity=80, ttype="Second")
        ]
        actual = align_to_start(notelist)
        for actual_note, expected_note in zip(actual, expected):
            # Velocity and pitch should be identical
            self.assertEqual(actual_note.pitch, expected_note.pitch)
            self.assertEqual(actual_note.velocity, expected_note.velocity)
            # Allow some wiggle room in duration and time
            self.assertAlmostEqual(actual_note.duration, expected_note.duration)
            self.assertAlmostEqual(actual_note.time, expected_note.time)

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
        actual = merge_repeated_notes(notelist, overlap_milliseconds=3)
        self.assertEqual(actual, expected)
        # Even these notes are adjacent, they shouldn't be merged as they are different pitches
        notelist = [
            Note(pitch=90, duration=5, time=110, velocity=80, ttype="tick"),
            Note(pitch=91, duration=5, time=120, velocity=80, ttype="tick"),
        ]
        expected = notelist
        actual = merge_repeated_notes(notelist, overlap_milliseconds=3)
        self.assertEqual(actual, expected)
        # These notes are not adjacent, so shouldn't be touched
        notelist = [
            Note(pitch=90, duration=5, time=110, velocity=80, ttype="tick"),
            Note(pitch=90, duration=5, time=1000, velocity=80, ttype="tick"),
        ]
        expected = notelist
        actual = merge_repeated_notes(notelist, overlap_milliseconds=3)
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
        actual = merge_repeated_notes(notelist, overlap_milliseconds=25)
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

    def test_remove_invalid_notes(self):
        notes = [
            # This note is valid
            Note(pitch=90, duration=5, time=110, velocity=80, ttype="tick"),
            # This note is too low
            Note(pitch=1, duration=10, time=1000, velocity=50, ttype="tick"),
            # This note is also too low
            Note(pitch=20, duration=10, time=1000, velocity=50, ttype="tick"),
            # This note is too high
            Note(pitch=125, duration=10, time=2000, velocity=50, ttype="tick"),
            # This note is also too high
            Note(pitch=109, duration=10, time=1000, velocity=50, ttype="tick"),
            # This note is valid
            Note(pitch=50, duration=10, time=3000, velocity=60, ttype="tick"),
        ]
        expected_len = 2
        actual_len = len(remove_out_of_range_notes(notes))
        self.assertEqual(expected_len, actual_len)

    def test_cap_long_notes(self):
        notes = [
            Note(pitch=90, duration=1.0, time=1., velocity=80, ttype="Second"),
            # This note will be capped to a five-second duration
            Note(pitch=1, duration=10, time=2., velocity=50, ttype="Second"),
            Note(pitch=20, duration=4., time=3., velocity=50, ttype="Second"),
        ]
        capped = cap_long_notes(notes, max_duration_milliseconds=5.)
        self.assertTrue(capped[0].duration == 1.)  # no cap
        self.assertTrue(capped[1].duration == 5.)  # capped
        self.assertTrue(capped[2].duration == 4.)  # no cap frfr
        # Repeat, but cap at 3 seconds
        capped_4 = cap_long_notes(notes, max_duration_milliseconds=3.)
        self.assertTrue(capped_4[0].duration == 1.)  # no cap
        self.assertTrue(capped_4[1].duration == 3.)
        self.assertTrue(capped_4[2].duration == 3.)

    def test_remove_overlap(self):
        notes = [
            # This note overlaps with the next one, so its duration will be capped
            Note(pitch=90, duration=1.5, time=1., velocity=80, ttype="Second"),
            Note(pitch=90, duration=1.0, time=2., velocity=50, ttype="Second"),
            Note(pitch=93, duration=0.5, time=2., velocity=50, ttype="Second"),
            Note(pitch=90, duration=2.0, time=4., velocity=50, ttype="Second"),
        ]
        no_overlap = sorted(remove_overlap(notes), key=lambda x: x.time)
        self.assertTrue(no_overlap[0].duration == 1.0)
        self.assertTrue(no_overlap[1].duration == 1.0)
        self.assertTrue(no_overlap[2].duration == 0.5)
        self.assertTrue(no_overlap[3].duration == 2.0)
        self.assertTrue(len(no_overlap) == len(notes))
        # No changes to any other parameters
        for newnote, oldnote in zip(no_overlap, notes):
            self.assertEqual(newnote.pitch, oldnote.pitch)
            self.assertEqual(newnote.time, oldnote.time)
            self.assertEqual(newnote.velocity, oldnote.velocity)

    def test_remove_duplicates(self):
        notes = [
            # We'll only keep one of the next two notes
            Note(pitch=90, duration=1.5, time=1., velocity=80, ttype="Second"),
            Note(pitch=90, duration=1.5, time=1., velocity=80, ttype="Second"),
            # Differs by pitch
            Note(pitch=91, duration=1.5, time=1., velocity=80, ttype="Second"),
            # Differs by onset
            Note(pitch=90, duration=1.5, time=1.1, velocity=81, ttype="Second"),
            # Differs by duration
            Note(pitch=90, duration=1.55, time=1., velocity=80, ttype="Second"),
            # Differs by velocity
            Note(pitch=90, duration=1.5, time=1., velocity=81, ttype="Second"),
        ]
        deduped = remove_duplicate_notes(notes)
        self.assertTrue(len(deduped) == len(notes) - 1)  # removing one duplicate

    @unittest.skipIf(os.getenv("REMOTE") == "true", "Skipping test on GitHub Actions")
    def test_preprocess_score_full_dataset(self):
        """Tests our preprocess_score function on the entire dataset. Only runs locally"""
        datasets = [
            "raw/bushgrafts",
            "raw/pijama",
            "raw/jtd",
            "raw/jja",
            "raw/pianist8",
            "pretraining/atepp"
        ]
        # Iterate over all datasets
        for ds in datasets:
            # Add the beginning of the filepath
            ds = os.path.join(utils.get_project_root(), "data", ds)
            for t in tqdm(os.listdir(ds), desc="Testing preprocessing on dataset {}".format(ds)):

                # Skip over e.g. .gitkeep files
                if not os.path.isdir(os.path.join(ds, t)):
                    continue
                midi_fp = os.path.join(ds, t, "piano_midi.mid")

                # Load score and preprocess
                loaded = load_score(midi_fp, as_seconds=True)
                score = preprocess_score(loaded, min_duration_milliseconds=50, max_duration_milliseconds=5000)

                # All notes should be within range of the piano
                min_pitch, max_pitch = utils.get_pitch_range(score)
                self.assertTrue(utils.MIDI_OFFSET <= min_pitch < utils.PIANO_KEYS + utils.MIDI_OFFSET)
                self.assertTrue(utils.MIDI_OFFSET <= max_pitch < utils.PIANO_KEYS + utils.MIDI_OFFSET)
                self.assertTrue(min_pitch <= max_pitch)

                # No notes should have a short duration
                smallest_duration = min(score.tracks[0].notes, key=lambda x: x.duration).duration
                self.assertTrue(smallest_duration >= 0.05)

                # Notes should be capped at a maximum duration
                longest_duration = max(score.tracks[0].notes, key=lambda x: x.duration).duration
                self.assertTrue(longest_duration <= 5.)

                # Should be no overlap between notes
                for pitch in range(utils.MIDI_OFFSET, utils.MIDI_OFFSET + utils.PIANO_KEYS + 1):
                    at_pitch = sorted([n for n in score.tracks[0].notes if n.pitch == pitch], key=lambda x: x.time)
                    for n1, n2 in zip(at_pitch, at_pitch[1:]):
                        self.assertTrue(round(n2.start, 3) >= round(n1.end, 3))

                # First note should start at 0 seconds
                first_note = score.tracks[0].notes[0]
                self.assertAlmostEqual(first_note.time, 0.)

                # Should be no duplicates
                seen = set()
                note_list = score.tracks[0].notes
                for note in note_list:
                    note_key = (note.pitch, note.time, note.duration, note.velocity)
                    seen.add(note_key)
                self.assertTrue(len(list(seen)) == len(note_list))


class LoadScoreTest(unittest.TestCase):
    def test_load_score_as_seconds(self):
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
            loaded = load_score(os.path.join(utils.get_project_root(), "tests/test_resources", file), as_seconds=True)
            # Sanity check that the score is correct
            self.assertTrue(isinstance(loaded, Score))
            # Should have ttype == Seconds
            self.assertTrue(isinstance(loaded.ttype, Second))
            # Should have the correct resolution
            self.assertTrue(loaded.ticks_per_quarter == utils.TICKS_PER_QUARTER)
            # Should have the correct tempo
            self.assertTrue(len(loaded.tempos) == 1)
            self.assertTrue(loaded.tempos[0].qpm == utils.TEMPO)
            # Should have the correct time signatures
            self.assertTrue(len(loaded.time_signatures) == 1)
            self.assertTrue(loaded.time_signatures[0].numerator == utils.TIME_SIGNATURE)

    def test_load_score_tsd(self):
        # Create the tokenizer
        TOKENIZER = TSD(
            TokenizerConfig(
                # This means that we will have exactly 100 evenly-spaced tokens per "bar"
                beat_res={(0, utils.TIME_SIGNATURE): 100 // utils.TIME_SIGNATURE}
            )
        )
        # The tokenizer vocab is a dictionary: we just want to get the TimeShift and duration tokens
        timeshifts = [m for m in TOKENIZER.vocab.keys() if "TimeShift" in m]
        n_timeshifts = len(timeshifts)  # 100 tokens at evenly-spaced 10ms increments between 0 - 1 second
        durations = [m for m in TOKENIZER.vocab.keys() if "Duration" in m]
        n_durations = len(durations)  # 100 tokens at evenly-spaced 10ms increments between 0 - 1 second
        assert n_timeshifts == 100 == n_durations
        # Get tokens corresponding to particular time values
        ts_token_10ms = timeshifts[0]
        ts_token_100ms = timeshifts[9]
        ts_token_1000ms = timeshifts[-1]
        dur_token_10ms = durations[0]
        dur_token_100ms = durations[9]
        dur_token_1000ms = durations[-1]
        # Create some dummy pretty MIDI notes with standard durations
        dummy_notes = [
            PMNote(start=0., end=1., pitch=80, velocity=50),  # 1 second duration
            PMNote(start=1., end=1.1, pitch=81, velocity=50),  # 100 millisecond duration
            PMNote(start=1.1, end=1.11, pitch=82, velocity=50),  # 10 millisecond duration
            # Two-second rest between onset times
            PMNote(start=3.1, end=4.1, pitch=83, velocity=50),  # 1 second duration
            PMNote(start=3.2, end=3.3, pitch=84, velocity=50),  # 100 millisecond duration, during previous note
            # 10-millisecond rest between onset times
            PMNote(start=3.21, end=3.22, pitch=85, velocity=50),  # 10 millisecond duration, during previous note
            # 100-millisecond rest between onset times
            PMNote(start=3.31, end=3.32, pitch=86, velocity=50)  # 10 millisecond duration, during previous note
            # Even though our note starting at 3.1 doesn't end for another ~700 ms
            #  We don't need any more tokens here, as we've already sent a duration token
        ]
        # Convert our dummy note stream into the expected tokens
        expected_tokens = [
            'Pitch_80', 'Velocity_51', dur_token_1000ms, ts_token_1000ms,
            'Pitch_81', 'Velocity_51', dur_token_100ms, ts_token_100ms,
            'Pitch_82', 'Velocity_51', dur_token_10ms,
            ts_token_1000ms, ts_token_1000ms,
            "Pitch_83", "Velocity_51", dur_token_1000ms, ts_token_100ms,
            "Pitch_84", "Velocity_51", dur_token_100ms, ts_token_10ms,
            "Pitch_85", "Velocity_51", dur_token_10ms, ts_token_100ms,
            "Pitch_86", "Velocity_51", dur_token_10ms
        ]
        # Create a fake MIDI file
        # Get a random tempo and time signature
        for _ in range(10):
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
            self.assertTrue(toks == expected_tokens)
            # Clean up
            os.remove(out_path)

    def test_load_score_midilike(self):
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
        for _ in range(10):
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
            # Sanity check that the symusic score is correct
            self.assertTrue(sm_load.ticks_per_quarter == utils.TICKS_PER_QUARTER)
            self.assertTrue(len(sm_load.tempos) == 1)
            self.assertTrue(sm_load.tempos[0].qpm == utils.TEMPO)
            self.assertTrue(len(sm_load.time_signatures) == 1)
            self.assertTrue(sm_load.time_signatures[0].numerator == utils.TIME_SIGNATURE)
            # Get the notes from both files and sort by onset time (not sure if this is necessary)
            pm_notes = sorted(get_notes_from_score(pm_load), key=lambda x: x.start)  # handles multiple tracks
            sm_notes = sorted(sm_load.tracks[0].notes, key=lambda x: x.start)
            # Number of notes should be the same
            self.assertTrue(len(pm_notes) == len(sm_notes))
            # Start times for notes should be directly equivalent
            for pm, sm in zip(pm_notes, sm_notes):
                self.assertTrue(utils.base_round(pm.start * 1000, 10) == sm.time)  # make sure to round!
                self.assertTrue(utils.base_round(pm.duration * 1000, 10) == sm.duration)

    @unittest.skipIf(os.getenv("REMOTE") == "true", "Skipping test on GitHub Actions")
    def test_load_score_full_dataset(self):
        """Tests our load_score function on the entire dataset. Only runs locally"""
        datasets = [
            "raw/bushgrafts",
            "raw/pijama",
            "raw/jtd",
            "raw/jja",
            "raw/pianist8",
            "pretraining/atepp"
        ]
        # Iterate over all datasets
        for ds in datasets:
            # Add the beginning of the filepath
            ds = os.path.join(utils.get_project_root(), "data", ds)
            for t in tqdm(os.listdir(ds), desc="Testing dataset {}".format(ds)):
                # Skip over e.g. .gitkeep files
                if not os.path.isdir(os.path.join(ds, t)):
                    continue
                # Check MIDI and metadata paths
                midi_fp = os.path.join(ds, t, "piano_midi.mid")
                # Load with Symusic and ttype=="Second" to get actual times
                pm_load = Score(midi_fp, ttype="Second")
                # Load with our custom symusic function
                sm_load = load_score(midi_fp)
                self.assertTrue(sm_load.ticks_per_quarter == utils.TICKS_PER_QUARTER)
                self.assertTrue(len(sm_load.tempos) == 1)
                self.assertTrue(sm_load.tempos[0].qpm == utils.TEMPO)
                self.assertTrue(len(sm_load.time_signatures) == 1)
                self.assertTrue(sm_load.time_signatures[0].numerator == utils.TIME_SIGNATURE)
                # Get the notes from both files and sort by onset time (not sure if this is necessary)
                pm_notes = sorted(get_notes_from_score(pm_load), key=lambda x: x.start)  # handles multiple tracks
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
