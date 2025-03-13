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
    TokTrainingIteratorAugmentation,
    load_or_train_tokenizer,
    DEFAULT_TOKENIZER_CLASS,
    DEFAULT_TOKENIZER_CONFIG
)


class TokenizerTest(unittest.TestCase):
    def test_getter(self):
        types = [REMI, TSD, Structured, PerTok, MIDILike]
        names = ["remi", "tsd", "structured", "pertok", "midilike"]
        for ty, name in zip(types, names):
            actual = get_tokenizer_class_from_string(name)
            self.assertEqual(actual, ty)

    def test_training_iterator(self):
        token_factory = REMI()
        fps = [os.path.join(utils.get_project_root(), "tests/test_resources/test_midi1/piano_midi.mid")]
        tti = TokTrainingIteratorAugmentation(token_factory, fps)
        # We should only have one track, but we should be augmenting it multiple times
        self.assertEqual(len(tti.files_paths), 1)
        self.assertGreater(len(tti), 1)
        # We shouldn't be splitting IDs, so we should only get one item per track
        token_factory.config.encode_ids_split = "no"
        gotter = tti.__getitem__(0)
        self.assertEqual(len(gotter), 1)
        # Now we can try with splitting
        token_factory.config.encode_ids_split = "bar"
        gotter = tti.__getitem__(0)
        self.assertGreater(len(gotter), 1)

    def test_load_or_train_tokenizer(self):
        fps = [os.path.join(utils.get_project_root(), "tests/test_resources/test_midi1/piano_midi.mid")]
        # Test without training
        actual = load_or_train_tokenizer("tmp.json", dict(do_training=False), fps)
        self.assertTrue(isinstance(actual, get_tokenizer_class_from_string(DEFAULT_TOKENIZER_CLASS)))
        self.assertFalse(actual.is_trained)
        # We should dump the tokenizer to a json
        self.assertTrue(os.path.isfile("tmp.json"))
        os.remove("tmp.json")

        # Test with training
        actual = load_or_train_tokenizer("tmp.json", dict(do_training=True), fps)
        self.assertTrue(isinstance(actual, get_tokenizer_class_from_string(DEFAULT_TOKENIZER_CLASS)))
        self.assertTrue(actual.is_trained)
        # We should dump the tokenizer to a json
        self.assertTrue(os.path.isfile("tmp.json"))
        os.remove("tmp.json")

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
        self.assertTrue("TEMPOCUSTOM_100" in tok.vocab.keys())
        self.assertTrue("TEMPOCUSTOM_150" in tok.vocab.keys())
        self.assertTrue("TEMPOCUSTOM_200" in tok.vocab.keys())

    def test_add_timesignatures(self):
        tok = MIDILike(TokenizerConfig(**DEFAULT_TOKENIZER_CONFIG))
        prev_vocab = tok.vocab_size
        # Adding three time signature tokens
        add_timesignatures_to_vocab(tok, [3, 4, 5])
        expected_vocab = prev_vocab + 3
        self.assertEqual(tok.vocab_size, expected_vocab)
        self.assertTrue("TIMESIGNATURECUSTOM_34" in tok.vocab.keys())
        self.assertTrue("TIMESIGNATURECUSTOM_44" in tok.vocab.keys())
        self.assertTrue("TIMESIGNATURECUSTOM_54" in tok.vocab.keys())

    def test_add_pianists(self):
        tok = MIDILike(TokenizerConfig(**DEFAULT_TOKENIZER_CONFIG))
        prev_vocab = tok.vocab_size
        # These are the files we have dummy metadata for
        files = [
            "test_midi1",
            "test_midi2",
            "test_midi3",
            "test_midi_bushgrafts1",
            "test_midi_bushgrafts2",
            "test_midi_bushgrafts3",
            "test_midi_jja1"
        ]
        files = [os.path.join(utils.get_project_root(), "tests/test_resources", f, "metadata_tivo.json") for f in files]
        # Add to the vocabulary using the metadata files we've defined
        add_pianists_to_vocab(tok, files)
        self.assertTrue(tok.vocab_size > prev_vocab)
        self.assertTrue(len(tok.vocab.keys()) == len(set(tok.vocab.keys())))  # should be no duplicates
        # These are the pianists we should be adding to our vocab
        expected_pianist_tokens = [
            "PIANIST_KennyBarron", "PIANIST_BeegieAdair", "PIANIST_BradMehldau", "PIANIST_HerbieHancock"
        ]
        for expect in expected_pianist_tokens:
            self.assertTrue(expect in tok.vocab)
        # We should not add the following pianists: they're in our EXCLUDE list
        not_expected = ["PIANIST_DougMcKenzie", "PIANIST_JJAPianist1"]
        for not_expect in not_expected:
            self.assertFalse(not_expect in tok.vocab)

    def test_add_genres(self):
        tok = MIDILike(TokenizerConfig(**DEFAULT_TOKENIZER_CONFIG))
        prev_vocab = tok.vocab_size
        # These are the files we have dummy metadata for
        files = [
            "test_midi1",
            "test_midi2",
            "test_midi3",
            "test_midi_bushgrafts1",
            "test_midi_bushgrafts2",
            "test_midi_bushgrafts3",
            "test_midi_jja1"
        ]
        files = [os.path.join(utils.get_project_root(), "tests/test_resources", f, "metadata_tivo.json") for f in files]
        # Add to the vocabulary using the metadata files we've defined
        add_genres_to_vocab(tok, files)
        self.assertTrue(tok.vocab_size == prev_vocab + 4)
        self.assertTrue(len(tok.vocab.keys()) == len(set(tok.vocab.keys())))  # should be no duplicates
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

    @unittest.skipIf(os.getenv("REMOTE") == "true", "Skipping test on GitHub Actions")
    def test_add_all_genres(self):
        tokfactory = REMI()
        js_fps = utils.get_data_files_with_ext("data/raw", "**/*_tivo.json")
        add_genres_to_vocab(tokfactory, js_fps)
        tok_genres = sorted(set([i for i in tokfactory.vocab.keys() if "GENRES" in i]))
        self.assertEqual(len(tok_genres), 26)

    @unittest.skipIf(os.getenv("REMOTE") == "true", "Skipping test on GitHub Actions")
    def test_add_all_pianists(self):
        tokfactory = REMI()
        js_fps = utils.get_data_files_with_ext("data/raw", "**/*_tivo.json")
        add_pianists_to_vocab(tokfactory, js_fps)
        tok_pianists = sorted(set([i for i in tokfactory.vocab.keys() if "PIANIST" in i]))
        self.assertEqual(len(tok_pianists), 129)  # pijama + JTD pianists


if __name__ == '__main__':
    unittest.main()
