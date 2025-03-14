#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for tokenizers"""

import os
import unittest

from miditok import REMI, TSD, Structured, PerTok, MIDILike, TokenizerConfig

from jazz_style_conditioned_generation import utils
from jazz_style_conditioned_generation.data.tokenizer import (
    add_tempos_to_vocab,
    add_timesignatures_to_vocab,
    add_pianists_to_vocab,
    add_genres_to_vocab,
    get_tokenizer_class_from_string,
    load_tokenizer,
    train_tokenizer,
    DEFAULT_TOKENIZER_CONFIG
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
        tok = load_tokenizer()
        self.assertTrue(isinstance(tok, TSD))
        self.assertFalse(tok.config.use_tempos)
        self.assertFalse(tok.config.use_pitch_bends)
        self.assertFalse(tok.config.use_time_signatures)
        self.assertFalse(tok.is_trained)
        # Test with a different tokenizer type
        tok = load_tokenizer(tokenizer_str="midilike")
        self.assertTrue(isinstance(tok, MIDILike))
        # Should have exactly 100 timeshift tokens
        ts_tokens = [t for t in tok.vocab if "TimeShift" in t]
        self.assertTrue(len(ts_tokens) == 100)

    def test_train_tokenizer(self):
        midi_files = [
            "test_midi1/piano_midi.mid",
            "test_midi2/piano_midi.mid",
            "test_midi3/piano_midi.mid",
        ]
        midi_files = [os.path.join(utils.get_project_root(), "tests/test_resources", mf) for mf in midi_files]
        # Get the tokenizer
        tok = load_tokenizer(tokenizer_str="midilike")
        # Train with 1000 vocab size and our three midi files
        train_tokenizer(tok, midi_files, vocab_size=1000)
        self.assertTrue(tok.is_trained)
        self.assertTrue(tok.vocab_size == 1000)
        # Should have exactly 100 timeshift tokens
        ts_tokens = [t for t in tok.vocab if "TimeShift" in t]
        self.assertTrue(len(ts_tokens) == 100)

    def test_default_tokenizer_config(self):
        tok = MIDILike(TokenizerConfig(**DEFAULT_TOKENIZER_CONFIG))
        # Expecting exactly 100 evenly-spaced timeshift tokens
        tshift_toks = [i for i in tok.vocab.keys() if "TimeShift" in i]
        self.assertTrue(len(tshift_toks) == 100)
        # Expecting 88 note-on/note-off tokens
        noteon_toks = [i for i in tok.vocab.keys() if "NoteOn" in i]
        self.assertTrue(len(noteon_toks) == utils.PIANO_KEYS)
        noteoff_toks = [i for i in tok.vocab.keys() if "NoteOff" in i]
        self.assertTrue(len(noteoff_toks) == utils.PIANO_KEYS)

    def test_add_tempos(self):
        tok = MIDILike(TokenizerConfig(**DEFAULT_TOKENIZER_CONFIG))
        prev_vocab = tok.vocab_size
        # Adding eleven tempo tokens -- [100, 110, 120, ..., 200]
        add_tempos_to_vocab(tok, (100, 200), 11)
        expected_vocab = prev_vocab + 11
        self.assertEqual(tok.vocab_size, expected_vocab)
        expected = ["TEMPOCUSTOM_100", "TEMPOCUSTOM_150", "TEMPOCUSTOM_200"]
        for expect in expected:
            self.assertTrue(expect in tok.vocab)
            # self.assertTrue(expect in tok.special_tokens)

    def test_add_timesignatures(self):
        tok = MIDILike(TokenizerConfig(**DEFAULT_TOKENIZER_CONFIG))
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
        tok = MIDILike(TokenizerConfig(**DEFAULT_TOKENIZER_CONFIG))
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
        tok = MIDILike(TokenizerConfig(**DEFAULT_TOKENIZER_CONFIG))
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


if __name__ == '__main__':
    unittest.main()
