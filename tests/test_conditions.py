#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for creating conditions"""

import os
import unittest

from miditok import REMI

from jazz_style_conditioned_generation import utils
from jazz_style_conditioned_generation.data import conditions as cond
from jazz_style_conditioned_generation.data.tokenizer import (
    add_pianists_to_vocab,
    add_genres_to_vocab,
    add_timesignatures_to_vocab,
    add_tempos_to_vocab
)


class ConditionsTest(unittest.TestCase):
    def test_validate_conditions(self):
        accept = ["genres", "moods"]
        self.assertIsNone(cond.validate_conditions(accept))
        reject = ["asdfasdga", "asgsdd"]
        self.assertRaises(ValueError, cond.validate_conditions, reject)

    def test_get_inner_json_values(self):
        metadata = {
            "key": "value",
            "nested_key": [
                {
                    "name": "nested_value1"
                },
                {
                    "name": "nested_value2"
                }
            ]
        }
        expected = ["value"]
        self.assertEqual(list(cond.get_inner_json_values(metadata, "key")), expected)
        expected2 = ["nested_value1", "nested_value2"]
        self.assertEqual(list(cond.get_inner_json_values(metadata, "nested_key")), expected2)

    def test_validate_condition_values(self):
        values = [
            ("African Folk", 3),  # merged into African
            ("Adult Alternative Pop/Rock", 8),  # merged into Pop/Rock
            ("African Folk", 5),  # merged into African, replaces the 3 weighting from African Folk
            ("Guitar Jazz", 9),  # Removed
            ("Modal Jazz", 3),  # Kept as is
        ]
        expected = [("Pop/Rock", 8), ("African", 5), ("Modal Jazz", 3)]
        actual = cond.validate_condition_values(values, "genres")
        self.assertEqual(actual, expected)

    def test_add_condition_tokens_to_sequence(self):
        dummy = [1, 3, 3, 3, 5, 6]
        condition_tokens = [100, 200, 300]
        # Condition tokens should be added before the start of the sequence, and it should be truncated to fit
        expected_inputs = [100, 200, 300, 1, 3, 3]
        expected_targets = [200, 300, 1, 3, 3, 3]
        actual_inputs, actual_targets = cond.add_condition_tokens_to_sequence(dummy, condition_tokens)
        self.assertEqual(expected_inputs, actual_inputs)
        self.assertEqual(expected_targets, actual_targets)
        # Testing with adding no condition tokens
        condition_tokens = []
        self.assertRaises(AssertionError, cond.add_condition_tokens_to_sequence, dummy, condition_tokens)

    def test_get_pianist_token(self):
        # Create the tokenizer and add to the vocabulary
        tokenizer = REMI()
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
        # Test without adding any tokens to the vocabulary
        with self.assertRaises(AssertionError):
            cond.get_pianist_tokens({}, tokenizer)
        # Add to the vocabulary using the metadata files we've defined
        add_pianists_to_vocab(tokenizer, files)
        # Test just getting the actual pianist from the track
        # This track is by Kenny Barron
        track = utils.read_json_cached(files[0])
        expected_token = ["PIANIST_KennyBarron"]
        actual_tokens = cond.get_pianist_tokens(track, tokenizer, n_pianists=1)
        self.assertEqual(expected_token, actual_tokens)
        self.assertTrue(len(actual_tokens) == 1)
        # This track is by Beegie Adair
        track = utils.read_json_cached(files[1])
        expected_token = ["PIANIST_BeegieAdair"]
        actual_tokens = cond.get_pianist_tokens(track, tokenizer, n_pianists=1)
        self.assertEqual(expected_token, actual_tokens)
        self.assertTrue(len(actual_tokens) == 1)
        # Test getting the actual pianist + top-1 most similar pianists
        # This track is by Kenny Barron, who is most similar to John Hicks
        track = utils.read_json_cached(files[0])
        expected_token = ["PIANIST_KennyBarron", "PIANIST_JohnHicks"]
        actual_tokens = cond.get_pianist_tokens(track, tokenizer, n_pianists=2)
        self.assertEqual(expected_token, actual_tokens)
        self.assertTrue(len(actual_tokens) == 2)
        # Test getting the actual pianist + top-2 most similar pianists
        # This track is by Beegie Adair, who is most similar to Cyrus Chestnut + Brad Mehldau
        track = utils.read_json_cached(files[1])
        expected_token = ["PIANIST_BeegieAdair", "PIANIST_CyrusChestnut", "PIANIST_BradMehldau"]
        actual_tokens = cond.get_pianist_tokens(track, tokenizer, n_pianists=3)
        self.assertEqual(expected_token, actual_tokens)
        self.assertTrue(len(actual_tokens) == 3)

    def test_get_genre_token(self):
        # Create the tokenizer and add to the vocabulary
        tokenizer = REMI()
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
        # Test without adding any tokens to the vocabulary
        with self.assertRaises(AssertionError):
            cond.get_genre_tokens({}, tokenizer)
        # Add to the vocabulary using the metadata files we've defined
        add_genres_to_vocab(tokenizer, files)
        # This track has genres associated with it directly
        track = utils.read_json_cached(files[0])
        expected_token = sorted(["GENRES_HardBop", "GENRES_PostBop", "GENRES_Caribbean"])
        actual_tokens = cond.get_genre_tokens(track, tokenizer, n_genres=None)
        self.assertEqual(expected_token, actual_tokens)
        self.assertTrue(len(actual_tokens) == 3)
        # Now, we can test by getting only the top-2 tokens with the strongest weighting
        expected_token = sorted(["GENRES_Caribbean", "GENRES_HardBop"])
        actual_tokens = cond.get_genre_tokens(track, tokenizer, n_genres=2)
        self.assertEqual(expected_token, actual_tokens)
        self.assertTrue(len(actual_tokens) == 2)
        # This track does not have genres associated with it, so we'll grab those associated with the pianist instead
        track = utils.read_json_cached(files[1])
        expected_token = ["GENRES_StraightAheadJazz"]  # Associated with Beegie Adair the artist, not this track
        actual_tokens = cond.get_genre_tokens(track, tokenizer)
        self.assertEqual(expected_token, actual_tokens)
        self.assertTrue(len(actual_tokens) == 1)
        # This track does not have any genres associated with it at all, so we should get an empty list
        track = utils.read_json_cached(files[-1])
        expected_token = []
        actual_tokens = cond.get_genre_tokens(track, tokenizer)
        self.assertEqual(expected_token, actual_tokens)

    def test_get_timesignature_token(self):
        # Create the tokenizer and add to the vocabulary
        tokenizer = REMI()
        # Should raise an error if we haven't added the time signature tokens in yet
        with self.assertRaises(AssertionError):
            _ = cond.get_time_signature_token(3, tokenizer)
        # Add in 3/4, 4/4, and 5/4 time signatures to the tokenizer vocabulary
        add_timesignatures_to_vocab(tokenizer, [3, 4, 5])
        # Test a 4/4 time signature
        ts = 4
        expected_token = "TIMESIGNATURECUSTOM_44"
        actual_token = cond.get_time_signature_token(ts, tokenizer)
        self.assertTrue(expected_token == actual_token)
        # Test a 3/4 time signature
        ts = 3
        expected_token = "TIMESIGNATURECUSTOM_34"
        actual_token = cond.get_time_signature_token(ts, tokenizer)
        self.assertTrue(expected_token == actual_token)
        # Test a 6/4 time signature: we haven't added this one, so should get an error
        with self.assertRaises(AttributeError):
            _ = cond.get_time_signature_token(6, tokenizer)

    def test_get_tempo_token(self):
        # Create the tokenizer and add to the vocabulary
        tokenizer = REMI()
        # Should raise an error if we haven't added the time signature tokens in yet
        with self.assertRaises(AssertionError):
            _ = cond.get_tempo_token(100, tokenizer)
        # Add in tempo tokens from [100, 110, 120, ..., 200]
        add_tempos_to_vocab(tokenizer, (100, 200), 11)
        # Test with 154 BPM, should be rounded to 150
        expected = "TEMPOCUSTOM_150"
        actual = cond.get_tempo_token(154, tokenizer)
        self.assertEqual(expected, actual)
        # Test with 156 BPM, should be rounded to 160
        expected = "TEMPOCUSTOM_160"
        actual = cond.get_tempo_token(156, tokenizer)
        self.assertEqual(expected, actual)
        # Raise an error if we try an out of range value
        with self.assertRaises(ValueError):
            _ = cond.get_tempo_token(20000, tokenizer)

    @unittest.skipIf(os.getenv("REMOTE") == "true", "Skipping test on GitHub Actions")
    def test_genre_tokens_with_full_dataset(self):
        tokfactory = REMI()
        js_fps = utils.get_data_files_with_ext("data/raw", "**/*_tivo.json")
        # Here, we're adding genre tokens from the ENTIRE dataset to our vocabulary
        add_genres_to_vocab(tokfactory, js_fps)
        track_genres = []
        # Now we simulate "getting" all the genre tokens for every track in the dataset
        for js in js_fps:
            js_loaded = utils.read_json_cached(js)
            track_genres.extend(cond.get_genre_tokens(js_loaded, tokfactory, n_genres=None))  # get all genres
        # We should have at least one appearance of every genre token we added to the tokenizer
        self.assertEqual(sorted(set(track_genres)), sorted(set([i for i in tokfactory.vocab.keys() if "GENRES" in i])))

    @unittest.skipIf(os.getenv("REMOTE") == "true", "Skipping test on GitHub Actions")
    def test_pianist_tokens_with_full_dataset(self):
        tokfactory = REMI()
        js_fps = utils.get_data_files_with_ext("data/raw", "**/*_tivo.json")
        self.assertTrue(len(js_fps) > 1000)  # should have a lot of files!
        # Here, we're adding pianist tokens from the ENTIRE dataset to our vocabulary
        add_pianists_to_vocab(tokfactory, js_fps)
        track_pianists = []
        # Now, we simulate "getting" all the pianist tokens for every track
        for js in js_fps:
            js_loaded = utils.read_json_cached(js)
            track_pianists.extend(cond.get_pianist_tokens(js_loaded, tokfactory, n_pianists=1))  # track pianist only
        tok_pianists = sorted(set([i for i in tokfactory.vocab.keys() if "PIANIST" in i]))
        # We should have at least one appearance of every pianist token we added to the tokenizer
        self.assertEqual(sorted(set(track_pianists)), tok_pianists)


if __name__ == '__main__':
    unittest.main()
