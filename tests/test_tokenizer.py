#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for tokenizers"""

import os
import unittest
from copy import deepcopy
from math import isclose

import numpy as np
import torch
from miditok import REMI, TSD, Structured, PerTok, MIDILike, TokenizerConfig, TokSequence
from symusic import Note, Score
from tqdm import tqdm

from jazz_style_conditioned_generation import utils
from jazz_style_conditioned_generation.data.scores import preprocess_score, load_score, note_list_to_score
from jazz_style_conditioned_generation.data.tokenizer import (
    add_tempos_to_vocab,
    add_timesignatures_to_vocab,
    add_pianists_to_vocab,
    add_genres_to_vocab,
    add_recording_years_to_vocab,
    get_tokenizer_class_from_string,
    load_tokenizer,
    train_tokenizer,
    DEFAULT_TOKENIZER_CONFIG,
    CustomTSD,
    CustomTokenizerConfig
)


class TokenizerTest(unittest.TestCase):
    def test_getter(self):
        types = [REMI, TSD, Structured, PerTok, MIDILike]
        names = ["remi", "tsd", "structured", "pertok", "midilike"]
        for ty, name in zip(types, names):
            actual = get_tokenizer_class_from_string(name)
            self.assertEqual(actual, ty)

    def test_load_tokenizer(self):
        # Test with all defaults
        tok = load_tokenizer(tokenizer_str="tsd")
        self.assertTrue(isinstance(tok, TSD))
        self.assertFalse(tok.config.use_tempos)
        self.assertFalse(tok.config.use_pitch_bends)
        self.assertFalse(tok.config.use_time_signatures)
        self.assertFalse(tok.is_trained)
        # Test that we've set our "BPE" token mapping correctly
        #  This should just go from token1 IDX -> [token1 IDX], token2 IDX -> [token2 IDX]
        #  We set it here to ensure compatibility between trained + non-trained tokenizers
        self.assertTrue(hasattr(tok, "bpe_token_mapping"))
        for k, v in tok.bpe_token_mapping.items():
            self.assertTrue(len(v) == 1)
            self.assertTrue(k == v[0])
        # Test with a different tokenizer type
        tok = load_tokenizer(tokenizer_str="midilike")
        self.assertTrue(isinstance(tok, MIDILike))
        # Should have exactly 100 timeshift tokens
        ts_tokens = [t for t in tok.vocab if "TimeShift" in t]
        self.assertTrue(len(ts_tokens) == 100)

    def test_load_custom_tokenizer(self):
        # Test with all defaults
        tok = load_tokenizer(tokenizer_str="custom-tsd")
        self.assertTrue(isinstance(tok, CustomTSD))
        self.assertFalse(tok.config.use_tempos)
        self.assertFalse(tok.config.use_pitch_bends)
        self.assertFalse(tok.config.use_time_signatures)
        self.assertFalse(tok.is_trained)
        # Should have some additional parameters vs a vanilla miditok tokenizer
        self.assertTrue(hasattr(tok.config, "time_range"))
        self.assertTrue(hasattr(tok.config, "time_factor"))
        # Should have 102 timeshift tokens
        ts_tokens = [t for t in tok.vocab if "TimeShift" in t]
        self.assertTrue(len(ts_tokens) == 102)

    def test_train_tokenizer(self):
        midi_files = [
            "test_midi1/piano_midi.mid",
            "test_midi2/piano_midi.mid",
            "test_midi3/piano_midi.mid",
        ]
        midi_files = [os.path.join(utils.get_project_root(), "tests/test_resources", mf) for mf in midi_files]
        # Get the tokenizer
        tok = load_tokenizer(tokenizer_str="tsd")
        # Train with 1000 vocab size and our three midi files
        train_tokenizer(tok, midi_files, vocab_size=1000)
        self.assertTrue(tok.is_trained)
        self.assertTrue(tok.vocab_size == 1000)
        # Should have exactly 100 timeshift tokens
        ts_tokens = [t for t in tok.vocab if "TimeShift" in t]
        self.assertTrue(len(ts_tokens) == 100)
        # We should have updated the BPE token mapping item
        self.assertTrue(hasattr(tok, "bpe_token_mapping"))
        self.assertTrue(len(tok.bpe_token_mapping) == tok.vocab_size)
        # All the VALUES should correspond to "base" tokens
        for v in tok.bpe_token_mapping.values():
            for id_ in v:
                try:
                    _ = tok[id_]
                except KeyError:
                    self.fail()
        # However, the KEYS should correspond to "BPE" tokens
        with self.assertRaises(KeyError):
            for k in tok.bpe_token_mapping.keys():
                _ = tok[k]
        # If we try to train a tokenizer with a vocabulary size smaller than what it currently has
        tok = load_tokenizer(tokenizer_str="tsd")
        tok_bpe_mapping = deepcopy(tok.bpe_token_mapping)
        train_tokenizer(tok, midi_files, vocab_size=-1)
        # We shouldn't update our BPE token mapping from the base
        self.assertEqual(tok_bpe_mapping, tok.bpe_token_mapping)

    def test_default_tokenizer_config(self):
        tok = TSD(TokenizerConfig(**DEFAULT_TOKENIZER_CONFIG))
        # Expecting exactly 100 evenly-spaced timeshift and duration tokens
        tshift_toks = [i for i in tok.vocab.keys() if "TimeShift" in i]
        self.assertTrue(len(tshift_toks) == 100)
        dur_toks = [i for i in tok.vocab.keys() if "Duration" in i]
        self.assertTrue(len(dur_toks) == 100)
        # Expecting 88 pitch tokens
        noteon_toks = [i for i in tok.vocab.keys() if "Pitch" in i]
        self.assertTrue(len(noteon_toks) == utils.PIANO_KEYS)
        # noteoff_toks = [i for i in tok.vocab.keys() if "NoteOff" in i]
        # self.assertTrue(len(noteoff_toks) == utils.PIANO_KEYS)

    def test_add_tempos(self):
        tok = TSD(TokenizerConfig(**DEFAULT_TOKENIZER_CONFIG))
        prev_vocab = tok.vocab_size
        # Adding eleven tempo tokens -- [100, 110, 120, ..., 200]
        add_tempos_to_vocab(tok, 80, 30, factor=1.05)
        expected_vocab = prev_vocab + 30
        self.assertEqual(tok.vocab_size, expected_vocab)
        expected = ["TEMPOCUSTOM_80", "TEMPOCUSTOM_84", "TEMPOCUSTOM_330"]
        for expect in expected:
            self.assertTrue(expect in tok.vocab)
            # self.assertTrue(expect in tok.special_tokens)

    def test_add_timesignatures(self):
        tok = TSD(TokenizerConfig(**DEFAULT_TOKENIZER_CONFIG))
        prev_vocab = tok.vocab_size
        # Adding three time signature tokens
        add_timesignatures_to_vocab(tok, [3, 4, 5])
        expected_vocab = prev_vocab + 3
        self.assertEqual(tok.vocab_size, expected_vocab)
        expected = ["TIMESIGNATURECUSTOM_34", "TIMESIGNATURECUSTOM_44", "TIMESIGNATURECUSTOM_54"]
        for expect in expected:
            self.assertTrue(expect in tok.vocab)
            # self.assertTrue(expect in tok.special_tokens)

    def test_add_pianists(self):
        tok = TSD(TokenizerConfig(**DEFAULT_TOKENIZER_CONFIG))
        prev_vocab = tok.vocab_size
        # Add to the vocabulary using the metadata files we've defined
        add_pianists_to_vocab(tok)
        self.assertTrue(tok.vocab_size > prev_vocab)
        self.assertTrue(len(tok.vocab.keys()) == len(set(tok.vocab.keys())))  # should be no duplicates
        # These are the pianists we should be adding to our vocab
        expected_pianist_tokens = [
            "PIANIST_KennyBarron", "PIANIST_BradMehldau", "PIANIST_HerbieHancock", "PIANIST_BudPowell"
        ]
        for expect in expected_pianist_tokens:
            self.assertTrue(expect in tok.vocab)
            # self.assertTrue(expect in tok.special_tokens)
        # We should not add the following pianists: they're in our EXCLUDE list
        not_expected = ["PIANIST_DougMcKenzie", "PIANIST_JJAPianist1", "PIANIST_BeegieAdair", "PIANIST_Hiromi"]
        for not_expect in not_expected:
            self.assertFalse(not_expect in tok.vocab)

    def test_add_genres(self):
        tok = TSD(TokenizerConfig(**DEFAULT_TOKENIZER_CONFIG))
        prev_vocab = tok.vocab_size
        # Add to the vocabulary using the metadata files we've defined
        add_genres_to_vocab(tok)
        self.assertGreater(tok.vocab_size, prev_vocab)
        self.assertTrue(len(tok.vocab.keys()) == len(set(tok.vocab.keys())))  # should be no duplicates
        # These are the genres we're expecting to add to the vocabulary
        expected_genres = [
            "GENRES_HardBop", "GENRES_PostBop", "GENRES_Caribbean",  # from test_midi1 track metadata
            "GENRES_StraightAheadJazz"  # from Beegie Adair's artist metadata
        ]
        for expected in expected_genres:
            self.assertTrue(expected in tok.vocab)
            # self.assertTrue(expected in tok.special_tokens)
        # These are the genres we're not expecting to add
        not_expected_genres = [
            # from Beegie Adair's artist metadata, but will be removed/merged with other genres
            "GENRES_Vocal", "GENRES_JazzInstrument", "GENRES_Jazz",
            "GENRES_Calypso"  # from Kenny Barron's artist metadata, will be merged with Caribbean
        ]
        for not_expected in not_expected_genres:
            self.assertFalse(not_expected in tok.vocab)

    def test_add_years(self):
        tok = TSD(TokenizerConfig(**DEFAULT_TOKENIZER_CONFIG))
        prev_vocab = tok.vocab_size
        # Adding 17 years from 1945 -> 2025 in 5 year steps
        add_recording_years_to_vocab(tok, 1945, 2025, 5)
        expected_vocab = prev_vocab + 17
        self.assertEqual(tok.vocab_size, expected_vocab)
        expected = ["RECORDINGYEAR_1945", "RECORDINGYEAR_1970", "RECORDINGYEAR_2025"]
        for expect in expected:
            self.assertTrue(expect in tok.vocab)

    @unittest.skipIf(os.getenv("REMOTE") == "true", "Skipping test on GitHub Actions")
    def test_add_all_genres(self):
        tokfactory = REMI()
        add_genres_to_vocab(tokfactory)
        tok_genres = sorted(set([i for i in tokfactory.vocab.keys() if "GENRES" in i]))
        self.assertEqual(len(tok_genres), 20)
        # tok_genres = sorted(set([i for i in tokfactory.special_tokens if "GENRES" in i]))
        # self.assertEqual(len(tok_genres), 26)

    @unittest.skipIf(os.getenv("REMOTE") == "true", "Skipping test on GitHub Actions")
    def test_add_all_pianists(self):
        tokfactory = REMI()
        add_pianists_to_vocab(tokfactory)
        tok_pianists = sorted(set([i for i in tokfactory.vocab.keys() if "PIANIST" in i]))
        self.assertEqual(len(tok_pianists), 25)  # only 25 pianists with more than 50 recordings
        # tok_pianists = sorted(set([i for i in tokfactory.special_tokens if "PIANIST" in i]))
        # self.assertEqual(len(tok_pianists), 129)  # pijama + JTD pianists


def test_scores_equivalent(a, b):
    # Get the raw + processed notes
    raw_notes = sorted(a.tracks[0].notes, key=lambda x: x.start)
    proc_notes = sorted(b.tracks[0].notes, key=lambda x: x.start)
    # Remove notes that are shorter than 1 ms (can't be tokenized)
    raw_notes = [i for i in raw_notes if i.duration > 0.001]
    # Should have the same number of notes
    if len(raw_notes) != len(proc_notes):
        return False
    # Iterate over sorted notes
    for note_raw, note_proc in zip(raw_notes, proc_notes):
        # Notes should be equivalent
        if not all([
            isclose(note_raw.start, note_proc.start, abs_tol=0.1),  # similar start times, to nearest 100 milliseconds
            isclose(note_raw.end, note_proc.end, abs_tol=0.1),  # similar end times
            isclose(note_raw.duration, note_proc.duration, abs_tol=0.1),  # similar durations
            note_raw.pitch == note_proc.pitch,  # IDENTICAL pitches
            abs(note_raw.velocity - note_proc.velocity) <= 4  # close velocities
        ]):
            return False
    return True


class CustomTSDTest(unittest.TestCase):
    """Test suite for custom tokenizer class using expressive performance timing"""

    @classmethod
    def setUpClass(cls):
        # MIDI files used in remote testing
        cls.test_midis = utils.get_data_files_with_ext("tests/test_resources", "**/*.mid")
        # Time ranges + factors for nonlinear / linear
        cls.time_range = (0.01, 1.0)
        cls.time_factor_nonlinear = 1.03
        cls.time_factor_linear = 1.0
        # Nonlinear spacing, as we'll actually do it
        cls.tok_actual = CustomTSD(
            CustomTokenizerConfig(
                **DEFAULT_TOKENIZER_CONFIG,
                time_range=(0.01, 2.5),
                time_factor=1.03
            )
        )
        # Nonlinear spacing between successive time/duration tokens
        cls.tok_cfg_nonlinear = CustomTokenizerConfig(
            **DEFAULT_TOKENIZER_CONFIG,
            time_range=cls.time_range,
            time_factor=cls.time_factor_nonlinear,
        )
        cls.tok_nonlinear = CustomTSD(cls.tok_cfg_nonlinear)
        # Linear spacing between successive time/duration tokens
        cls.tok_cfg_linear = CustomTokenizerConfig(
            **DEFAULT_TOKENIZER_CONFIG,
            time_range=cls.time_range,
            time_factor=cls.time_factor_linear,
        )
        cls.tok_linear = CustomTSD(cls.tok_cfg_linear)
        # Vanilla MIDITok TSD tokenizer, without performance timing
        cls.tok_vanilla = TSD(CustomTokenizerConfig(**DEFAULT_TOKENIZER_CONFIG))

    def test_save_load_tokenizer(self):
        for num, tok_cls in enumerate([self.tok_actual, self.tok_linear, self.tok_nonlinear]):
            with self.subTest(tok_cls=tok_cls):
                # Copy the tokenizer
                tok = deepcopy(tok_cls)
                # Dump the tokenizer to a JSON
                outpath = os.path.join(utils.get_project_root(), f"tests/tokenizer_{num}.json")
                tok.save(outpath)
                self.assertTrue(os.path.isfile(outpath))
                # Now, try reloading the tokenizer
                newtok = CustomTSD(params=outpath)
                # Check that the tokenizers are equivalent
                self.assertEqual(newtok, tok)
                self.assertEqual(len(newtok), len(tok))
                # Tidy up by removing the saved JSON
                os.remove(outpath)

    def test_base_vocabulary(self):
        # test both linear and non-linear tokenizer classes
        for tok_cls in [self.tok_linear, self.tok_nonlinear]:
            with self.subTest(tok_cls=tok_cls):
                # Should have 88 pitch tokens (1 per piano key)
                pitch_tokens = [i for i in tok_cls.vocab.keys() if i.startswith("Pitch")]
                self.assertTrue(len(pitch_tokens) == utils.PIANO_KEYS)
                # Should have 32 velocity tokens, as specified in config
                velocity_tokens = [i for i in tok_cls.vocab.keys() if i.startswith("Velocity")]
                self.assertTrue(len(velocity_tokens) == 32)
                # Should have multiple duration + timeshift tokens
                timeshift_tokens = [i for i in tok_cls.vocab.keys() if i.startswith("TimeShift")]
                duration_tokens = [i for i in tok_cls.vocab.keys() if i.startswith("Duration")]
                self.assertTrue(len(timeshift_tokens) == len(duration_tokens))
                self.assertGreater(len(timeshift_tokens), 0)
                self.assertGreater(len(duration_tokens), 0)
                # Timeshift and duration tokens should be rounded to nearest 10ms
                for tok_type in [timeshift_tokens, duration_tokens]:
                    for tok in tok_type:
                        tok_val = float(tok.split("_")[-1])
                        self.assertTrue(tok_val == round(tok_val, 2))

    def test_special_tokens(self):
        # test both linear and non-linear tokenizer classes
        for tok_cls in [self.tok_linear, self.tok_nonlinear]:
            with self.subTest(tok_cls=tok_cls):
                # Should have BOS, PAD, and EOS tokens
                for token in ["BOS_None", "PAD_None", "EOS_None"]:
                    self.assertTrue(token in tok_cls.vocab)
                # Special tokens should be inserted earlier in the vocabulary than non-special tokens
                for token in ["Pitch", "Duration", "Velocity", "TimeShift"]:
                    compatible_tokens = [v for k, v in tok_cls.vocab.items() if k.startswith(token)]
                    smallest_token = min(compatible_tokens)
                    for special_token in ["BOS_None", "PAD_None", "EOS_None"]:
                        self.assertTrue(tok_cls.vocab[special_token] < smallest_token)
                # Special tokens should be identical to those in a vanilla MidiTok TSD
                self.assertTrue(self.tok_vanilla.special_tokens == tok_cls.special_tokens)

    def test_times_nonlinear(self):
        # Maximum and minimum times should be set as expected
        self.assertAlmostEqual(min(self.tok_nonlinear.times), round(self.time_range[0] * 1000))
        self.assertAlmostEqual(max(self.tok_nonlinear.times), round(self.time_range[1] * 1000))
        # Distance between all successive time values should not be the same
        diffs = np.diff(self.tok_nonlinear.times)
        self.assertFalse(np.allclose(diffs, diffs[0], atol=1e-4))
        # Difference between the last few time values should be greater than between the first
        for i in range(1, 11):
            self.assertGreater(diffs[-i], diffs[i])
        # Array should be sorted
        self.assertTrue(np.all(self.tok_nonlinear.times[:-1] <= self.tok_nonlinear.times[1:]))

    def test_times_linear(self):
        # Maximum and minimum times should be set as expected
        self.assertAlmostEqual(min(self.tok_linear.times), round(self.time_range[0] * 1000))
        self.assertAlmostEqual(max(self.tok_linear.times), round(self.time_range[1] * 1000))
        # Distance between all successive time values should be roughly the same
        diffs = np.diff(self.tok_linear.times)
        self.assertTrue(np.allclose(diffs, diffs[0], atol=1e-4))
        # Array should be sorted
        self.assertTrue(np.all(self.tok_nonlinear.times[:-1] <= self.tok_nonlinear.times[1:]))

    def test_versus_vanilla_miditok(self):
        test_midis = utils.get_data_files_with_ext("tests/test_resources", "**/*.mid")
        for midi in test_midis:
            # Load with our hack, then tokenize with vanilla tokenizer
            midi_vanilla = load_score(midi, as_seconds=False)
            tok_vanilla = self.tok_vanilla.encode(midi_vanilla, no_preprocess_score=False)
            # Load WITHOUT our hack, then tokenize with new tokenizer
            midi_new = load_score(midi, as_seconds=True)
            tok_new = self.tok_actual.encode(midi_new, no_preprocess_score=True)
            # Pitch and velocity tokens should be identical
            for token in ["Pitch", "Velocity"]:
                tokens_vanilla = [i for i in tok_vanilla[0].tokens if i.startswith(token)]
                tokens_new = [i for i in tok_new[0].tokens if i.startswith(token)]
                # Sorting as our tokenizer uses a different method for representing time than MIDITok
                # this means that notes in the token stream can occasionally appear in a different order
                # however, the de-tokenized score should still be identical
                self.assertEqual(sorted(tokens_vanilla), sorted(tokens_new))
            # Detokenize both token streams, and convert vanilla to Second type (from Tick)
            detok_vanilla = self.tok_vanilla.decode(tok_vanilla).to("Second")
            detok_new = self.tok_actual.decode(tok_new)
            # Get notes
            raw_notes = sorted(
                [i for i in detok_vanilla.tracks[0].notes if round(i.duration, 2) > 0.05],
                key=lambda x: (x.pitch, x.start, x.duration)
            )
            proc_notes = sorted(
                [i for i in detok_new.tracks[0].notes if round(i.duration, 2) > 0.05],
                key=lambda x: (x.pitch, x.start, x.duration,)
            )
            # Should have the same number of notes
            self.assertTrue(len(raw_notes) == len(proc_notes))
            # Iterate over sorted notes
            for note_raw, note_proc in zip(raw_notes, proc_notes):
                # Remember, the vanilla tokenizer caps out at 1 second duration, so these notes won't be equivalent!
                if note_proc.duration > 1.:
                    continue
                # Otherwise, duration and onset/offset should be approximately equal
                self.assertTrue(isclose(note_raw.start, note_proc.start, abs_tol=0.1))
                self.assertTrue(isclose(note_raw.end, note_proc.end, abs_tol=0.1))
                self.assertTrue(isclose(note_raw.duration, note_proc.duration, abs_tol=0.1))
                # Velocity and pitch should be exactly equal
                self.assertTrue(note_raw.pitch == note_proc.pitch)
                self.assertTrue(note_raw.velocity == note_proc.velocity)

    @unittest.skipIf(os.getenv("REMOTE") == "true", "Skipping test on GitHub Actions")
    def test_dummy_versus_vanilla_miditok(self):
        # Create two identical note lists: one as seconds, one as "fake ticks" (where 1ms == 1 tick)
        notelist_secs = [
            Note(pitch=81, duration=0.1, time=0.1, velocity=80, ttype="Second"),
            Note(pitch=82, duration=0.2, time=0.2, velocity=70, ttype="Second"),
            Note(pitch=83, duration=0.3, time=0.3, velocity=60, ttype="Second"),
            Note(pitch=84, duration=0.01, time=0.4, velocity=60, ttype="Second"),
            Note(pitch=85, duration=1.0, time=0.5, velocity=50, ttype="Second"),
            Note(pitch=82, duration=0.02, time=0.51, velocity=50, ttype="Second")
        ]
        notelist_ticks = [
            Note(pitch=81, duration=100, time=100, velocity=80, ttype="Tick"),
            Note(pitch=82, duration=200, time=200, velocity=70, ttype="Tick"),
            Note(pitch=83, duration=300, time=300, velocity=60, ttype="Tick"),
            Note(pitch=84, duration=10, time=400, velocity=60, ttype="Tick"),
            Note(pitch=85, duration=1000, time=500, velocity=50, ttype="Tick"),
            Note(pitch=82, duration=20, time=510, velocity=50, ttype="Tick")
        ]
        # Convert both to scores
        score_secs = note_list_to_score(notelist_secs, utils.TICKS_PER_QUARTER, ttype="Second")
        score_ticks = note_list_to_score(notelist_ticks, utils.TICKS_PER_QUARTER, ttype="Tick")
        # Sanity check
        self.assertTrue(score_secs.tpq == score_ticks.tpq)
        # Encode both
        tok_secs = self.tok_linear.encode(score_secs, no_preprocess_score=False)
        tok_ticks = self.tok_vanilla.encode(score_ticks, no_preprocess_score=False)
        self.assertTrue(type(tok_secs[0]) is type(tok_ticks[0]) is TokSequence)  # same type
        # Check the tokens are equivalent
        tok_secs = tok_secs[0].tokens
        tok_ticks = tok_ticks[0].tokens
        self.assertTrue(len(tok_secs) == len(tok_ticks))
        # Calculate time tokens from vanilla tokenizer
        time_toks = {k: v for k, v in self.tok_vanilla.vocab.items() if "Duration" in k}
        min_time_tok = time_toks[min(time_toks, key=time_toks.get)]
        time_toks = {k: v - min_time_tok for k, v in time_toks.items()}
        # Iterate over both tokens
        for tsec, ttick in zip(tok_secs, tok_ticks):
            # Tokens should be the same type
            tsec_type, tsec_val = tsec.split("_")
            ttick_type, ttick_val = ttick.split("_")
            self.assertTrue(tsec_type == ttick_type)
            # Pitch and velocity tokens should be identical
            if tsec_type == "Pitch" or tsec_type == "Velocity":
                self.assertTrue(tsec == ttick)
            # Duration and timeshift tokens need to be handled differently
            else:
                # Getting "raw" time back from the tick
                ttick_val = (time_toks[ttick.replace("TimeShift", "Duration")] + 1) * 10
                self.assertEqual(int(tsec_val), int(ttick_val))

    def test_encoded_decoded_scores_equivalent(self):
        # We can check both tokenizers here
        for tok_cls in [self.tok_actual, self.tok_linear, self.tok_nonlinear]:
            with self.subTest(tok_cls=tok_cls):
                for midi in self.test_midis:
                    # Load up the MIDI with default settings in symusic
                    raw = load_score(midi, as_seconds=True)
                    raw = preprocess_score(raw)
                    # Encode -> decode with the tokenizer
                    encoded = tok_cls.encode(raw)  # need to pass in the preprocessed score object
                    decoded = tok_cls.decode(encoded)
                    # Check the scores are equivalent
                    self.assertTrue(test_scores_equivalent(raw, decoded))

    @unittest.skipIf(os.getenv("REMOTE") == "true", "Skipping test on GitHub Actions")
    def test_encoded_decoded_scores_equivalent_all(self):
        # Load up all the MIDI paths inside the data directory
        all_midis_jazz = utils.get_data_files_with_ext("data/raw", "**/*.mid")
        all_midis_classical = utils.get_data_files_with_ext("data/pretraining", "**/*.mid")
        all_midis = (all_midis_jazz + all_midis_classical)
        # We can check both tokenizers here
        for tok_cls in [self.tok_actual]:
            for midi in tqdm(all_midis, desc=f"Testing cls {tok_cls}"):
                # Load up the MIDI with default settings in symusic
                raw = load_score(midi, as_seconds=True)
                raw = preprocess_score(raw)
                # Encode -> decode with the tokenizer
                encoded = tok_cls.encode(raw)  # need to pass in the preprocessed score object
                decoded = tok_cls.decode(encoded)
                # Check the scores are equivalent
                passed = test_scores_equivalent(raw, decoded)
                # TODO: failing
                self.assertTrue(passed)

    def test_durations_longer_than_max(self):
        # Test with a dummy example
        _, max_duration = self.time_range
        notelist = [
            # Note should decode to multiple duration tokens
            Note(pitch=80, duration=max_duration + (max_duration / 2), time=0., velocity=80, ttype="Second"),
        ]
        # Convert the notelist to a score with time in seconds
        score = note_list_to_score(notelist, utils.TICKS_PER_QUARTER, ttype="Second")
        for tok_cls in [self.tok_linear, self.tok_nonlinear]:
            with self.subTest(tok_cls=tok_cls):
                # Encode and grab tokens
                encoded = tok_cls.encode(score, no_preprocess_score=True)
                tokens = encoded[0].tokens
                # Get only the duration tokens: we should have more than 1, as the note is longer than our max
                duration_tokens = [i for i in tokens if i.startswith("Duration")]
                self.assertGreater(len(duration_tokens), 1)
                # Bonus: should not be any timeshift tokens
                timeshift_tokens = [i for i in tokens if i.startswith("TimeShift")]
                self.assertTrue(len(timeshift_tokens) == 0)

    def test_durations_shorter_than_max(self):
        # Test with a dummy example
        min_duration, _ = self.time_range
        notelist = [
            # Note should decode to a single duration token
            Note(pitch=80, duration=min_duration, time=0., velocity=80, ttype="Second"),
        ]
        # Convert the notelist to a score with time in seconds
        score = note_list_to_score(notelist, utils.TICKS_PER_QUARTER, ttype="Second")
        for tok_cls in [self.tok_linear, self.tok_nonlinear]:
            with self.subTest(tok_cls=tok_cls):
                # Encode and grab tokens
                encoded = tok_cls.encode(score, no_preprocess_score=True)
                tokens = encoded[0].tokens
                # Get only the duration tokens: we should have exactly one, as the note is our minimum value
                duration_tokens = [i for i in tokens if i.startswith("Duration")]
                self.assertEqual(len(duration_tokens), 1)

    def test_timeshifts_longer_than_max(self):
        # Test with a dummy example
        _, max_duration = self.time_range
        notelist = [
            # Note should decode to multiple duration tokens
            Note(pitch=80, duration=max_duration, time=max_duration + (max_duration / 2), velocity=80, ttype="Second"),
        ]
        # Convert the notelist to a score with time in seconds
        score = note_list_to_score(notelist, utils.TICKS_PER_QUARTER, ttype="Second")
        for tok_cls in [self.tok_linear, self.tok_nonlinear]:
            with self.subTest(tok_cls=tok_cls):
                # Encode and grab tokens
                encoded = tok_cls.encode(score, no_preprocess_score=True)
                tokens = encoded[0].tokens
                # Get only the timeshift tokens: we should have more than 1, as the note is longer than our max
                timeshift_tokens = [i for i in tokens if i.startswith("TimeShift")]
                self.assertGreater(len(timeshift_tokens), 1)
                # Bonus: timeshift token should be equivalent to our max
                duration_tokens = [i for i in tokens if i.startswith("Duration")][0]
                duration_val = int(duration_tokens.split("_")[-1])
                self.assertTrue(duration_val == max(tok_cls.times) == max_duration * 1000)

    def test_timeshifts_shorter_than_max(self):
        # Test with a dummy example
        min_duration, _ = self.time_range
        notelist = [
            # Note should decode to multiple duration tokens
            Note(pitch=80, duration=1.0, time=min_duration, velocity=80, ttype="Second"),
        ]
        # Convert the notelist to a score with time in seconds
        score = note_list_to_score(notelist, utils.TICKS_PER_QUARTER, ttype="Second")
        for tok_cls in [self.tok_linear, self.tok_nonlinear]:
            with self.subTest(tok_cls=tok_cls):
                # Encode and grab tokens
                encoded = tok_cls.encode(score, no_preprocess_score=True)
                tokens = encoded[0].tokens
                # Get only the timeshift tokens: we should have exactly 1, as the note is our minimum encoded duration
                timeshift_tokens = [i for i in tokens if i.startswith("TimeShift")]
                self.assertEqual(len(timeshift_tokens), 1)

    def test_dummy_midi_nonlinear(self):
        notelist = [
            Note(pitch=80, duration=0.567, time=0.2, velocity=70, ttype="Second"),  # one duration token
            Note(pitch=82, duration=2.0, time=0.3, velocity=80, ttype="Second"),  # one duration token
            # Duration greater than maximum: should have two duration tokens
            Note(pitch=84, duration=3.0, time=0.45, velocity=75, ttype="Second"),
            Note(pitch=94, duration=0.02, time=0.5, velocity=60, ttype="Second"),
            # Timeshift greater than maximum: should have two timeshift tokens
            Note(pitch=93, duration=0.5560, time=4.0, velocity=60, ttype="Second")
        ]
        # Convert the notelist to a score with time in seconds
        score = note_list_to_score(notelist, utils.TICKS_PER_QUARTER, ttype="Second")
        expected = [
            "TimeShift_200", "Pitch_80", "Velocity_71", "Duration_562",
            "TimeShift_100", "Pitch_82", "Velocity_79", "Duration_2003",
            "TimeShift_150", "Pitch_84", "Velocity_75", "Duration_2500", "Duration_499",
            "TimeShift_50", "Pitch_94", "Velocity_59", "Duration_20",
            "TimeShift_2500", "TimeShift_985", "TimeShift_10", "Pitch_93", "Velocity_59", "Duration_562"
        ]
        # Convert to tokens
        encoded = self.tok_actual.encode(score, no_preprocess_score=True)
        actual = encoded[0].tokens
        self.assertEqual(actual, expected)

    def test_dummy_midi_linear(self):
        notelist = [
            # First four notes have 1 timeshift, 1 duration token each
            Note(pitch=80, duration=0.5257, time=0.1, velocity=70, ttype="Second"),
            Note(pitch=82, duration=0.6, time=0.2, velocity=80, ttype="Second"),
            Note(pitch=84, duration=0.357, time=0.5, velocity=60, ttype="Second"),
            Note(pitch=86, duration=0.4206, time=0.66178, velocity=50, ttype="Second"),
            # This has two duration tokens, 1 timeshift token
            Note(pitch=70, duration=1.15, time=0.77178, velocity=50, ttype="Second"),
            # This has 2 timeshift tokens, 2 duration tokens
            Note(pitch=69, duration=2.5, time=1.88178, velocity=50, ttype="Second"),
            # This has 4 timeshift tokens, 4 duration tokens
            Note(pitch=77, duration=3.187920934, time=4.99178, velocity=50, ttype="Second")
        ]
        # Convert the notelist to a score with time in seconds
        score = note_list_to_score(notelist, utils.TICKS_PER_QUARTER, ttype="Second")
        expected = [  # This was calculated manually, without checking vs the actual output!!
            "TimeShift_100", "Pitch_80", "Velocity_71", "Duration_530",
            "TimeShift_100", "Pitch_82", "Velocity_79", "Duration_600",
            "TimeShift_300", "Pitch_84", "Velocity_59", "Duration_360",
            "TimeShift_160", "Pitch_86", "Velocity_51", "Duration_420",
            "TimeShift_110", "Pitch_70", "Velocity_51", "Duration_1000", "Duration_150",
            "TimeShift_1000", "TimeShift_110", "Pitch_69", "Velocity_51", "Duration_1000", "Duration_1000",
            "Duration_500", "TimeShift_1000", "TimeShift_1000", "TimeShift_1000", "TimeShift_110",
            "Pitch_77", "Velocity_51", "Duration_1000", "Duration_1000", "Duration_1000", "Duration_190"
        ]
        # Convert to tokens
        encoded = self.tok_linear.encode(score, no_preprocess_score=True)
        actual = encoded[0].tokens
        self.assertEqual(actual, expected)

    @unittest.skipIf(os.getenv("REMOTE") == "true", "Skipping test on GitHub Actions")
    def test_add_condition_tokens_to_vocab(self):
        for tok_cls in [self.tok_actual, self.tok_linear, self.tok_nonlinear]:
            with self.subTest(tok_cls=tok_cls):
                # Make a copy so we don't add anything to the underlying object
                tok = deepcopy(tok_cls)
                # Get the previous vocabulary size
                prev_vocab = tok.vocab_size
                # Add all the condition tokens to the class
                add_genres_to_vocab(tok)
                add_pianists_to_vocab(tok)
                add_recording_years_to_vocab(tok, 1945, 2025, step=5)  # [1945, 1950, ..., 2025]
                add_tempos_to_vocab(tok, 80, 30, factor=1.05)
                add_timesignatures_to_vocab(tok, [3, 4])
                # Vocabulary should have increased
                self.assertGreater(tok.vocab_size, prev_vocab)

                # These are the genres we're expecting to add to the vocabulary
                expected_genres = [
                    "GENRES_HardBop", "GENRES_PostBop", "GENRES_Caribbean",  # from test_midi1 track metadata
                    "GENRES_StraightAheadJazz"  # from Beegie Adair's artist metadata
                ]
                for expected in expected_genres:
                    self.assertTrue(expected in tok.vocab)
                # These are the genres we're not expecting to add
                not_expected_genres = [
                    # from Beegie Adair's artist metadata, but will be removed/merged with other genres
                    "GENRES_Vocal", "GENRES_JazzInstrument", "GENRES_Jazz",
                    "GENRES_Calypso"  # from Kenny Barron's artist metadata, will be merged with Caribbean
                ]
                for not_expected in not_expected_genres:
                    self.assertFalse(not_expected in tok.vocab)

                # These are the pianists we should be adding to our vocab
                expected_pianist_tokens = [
                    "PIANIST_KennyBarron", "PIANIST_BradMehldau", "PIANIST_HerbieHancock", "PIANIST_BudPowell"
                ]
                for expect in expected_pianist_tokens:
                    self.assertTrue(expect in tok.vocab)
                # We should not add the following pianists: they're in our EXCLUDE list
                not_expected = ["PIANIST_DougMcKenzie", "PIANIST_JJAPianist1", "PIANIST_BeegieAdair", "PIANIST_Hiromi"]
                for not_expect in not_expected:
                    self.assertFalse(not_expect in tok.vocab)

                # Check recording years added to vocabulary
                expected = ["RECORDINGYEAR_1945", "RECORDINGYEAR_1970", "RECORDINGYEAR_2025"]
                for expect in expected:
                    self.assertTrue(expect in tok.vocab)

                # Check time signatures
                expected = ["TIMESIGNATURECUSTOM_34", "TIMESIGNATURECUSTOM_44", ]
                for expect in expected:
                    self.assertTrue(expect in tok.vocab)

                # Check tempi
                expected = ["TEMPOCUSTOM_80", "TEMPOCUSTOM_84", "TEMPOCUSTOM_330"]
                for expect in expected:
                    self.assertTrue(expect in tok.vocab)

    def test_getitem(self):
        for tok_cls in [self.tok_actual, self.tok_linear, self.tok_nonlinear]:
            with self.subTest(tok_cls=tok_cls):
                # With a string input, we expect to receive an ID (int)
                out = tok_cls["Pitch_80"]
                self.assertIsInstance(out, int)
                # With an int input, we expect to receive a token (str)
                out = tok_cls[50]
                self.assertIsInstance(out, str)
                # Cannot parse tuple inputs
                with self.assertRaises(NotImplementedError):
                    _ = tok_cls[(1, 15)]
                # Check pad token ID as well
                self.assertTrue(tok_cls.pad_token_id == tok_cls["PAD_None"])

    def test_call(self):
        # Passing a path: should expect to receive a TokSequence
        test_midi = os.path.join(utils.get_project_root(), "tests/test_resources/test_midi1/piano_midi.mid")
        output = self.tok_actual(test_midi)
        self.assertIsInstance(output, list)
        self.assertIsInstance(output[0], TokSequence)
        # Passing a score: should also expect to receive a TokSequence
        test_score = preprocess_score(load_score(test_midi, as_seconds=True))
        output = self.tok_actual(test_score)
        self.assertIsInstance(output, list)
        self.assertIsInstance(output[0], TokSequence)
        # Passing a TokSequence, should expect to receive a score
        # We can just reuse the output from the previous call
        output_score = self.tok_actual(output)
        self.assertIsInstance(output_score, Score)
        # Passing a list of integers, should also get a score
        test_input = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 100, ]]
        output_score = self.tok_actual(test_input)
        self.assertIsInstance(output_score, Score)
        # Passing a numpy array, should also get a score
        test_input = np.array(test_input)
        output_score = self.tok_actual(test_input)
        self.assertIsInstance(output_score, Score)
        # Passing a tensor, should also get a score
        test_input = torch.tensor(test_input)
        output_score = self.tok_actual(test_input)
        self.assertIsInstance(output_score, Score)

    def test_len(self):
        for tok_cls in [self.tok_actual, self.tok_linear, self.tok_nonlinear]:
            with self.subTest(tok_cls=tok_cls):
                # All of these functions should return the same thing
                self.assertTrue(len(tok_cls) == tok_cls.len == tok_cls.vocab_size)
                # We expect a good number of tokens
                self.assertGreater(len(tok_cls), utils.PIANO_KEYS)


if __name__ == '__main__':
    utils.seed_everything(utils.SEED)
    unittest.main()
