#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for creating conditions"""

import unittest

from miditok import REMI

from jazz_style_conditioned_generation.data import conditions as cond
from jazz_style_conditioned_generation.data.tokenizer import (
    add_conditions_to_vocab,
    add_timesignatures_to_vocab,
    add_tempos_to_vocab
)

TEST_DICT = [
    {
        "artist_genres": [
            {
                "name": "artist_genre1"
            },
            {
                "name": "artist_genre2"
            }
        ],
        "album_genres": [
            {
                "name": "album_genre1"
            },
            {
                "name": "album_genre2"
            },
            {
                "name": "Jazz"
            }
        ]
    },
    {
        "artist_genres": [
            {
                "name": "artist_genre1"
            },
            {
                "name": "artist_genre3",
            },
            {
                "name": "Jazz Instrument",
            }
        ]
    }
]


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

    def test_get_condition_tokens(self):
        condition = "genres"
        expected_keys = [
            "artist_genre1",
            "artist_genre2",
            "artist_genre3",
            "album_genre1",
            "album_genre2",
        ]
        expected_values = [
            "GENRES_artistgenre1",
            "GENRES_artistgenre2",
            "GENRES_artistgenre3",
            "GENRES_albumgenre1",
            "GENRES_albumgenre2",
        ]
        actual = cond.get_condition_special_tokens(condition, TEST_DICT)
        self.assertEqual(sorted(list(actual.keys())), sorted(expected_keys))
        self.assertEqual(sorted(list(actual.values())), sorted(expected_values))

    def test_validate_condition_values(self):
        values = ["African Folk", "Adult Alternative Pop/Rock", "African Folk", "Guitar Jazz"]
        expected = ["African", "Pop/Rock"]
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

    def test_get_condition_tokens_for_track(self):
        # Create fake mapping of genres -> token strings
        condition_mapping = {
            "pianist": {
                "Billy Bob": "PIANIST_BillyBob",
            },
            'genres': {
                'African': 'GENRES_African',
                'Afro-Cuban Jazz': 'GENRES_AfroCubanJazz',
                'Asian': 'GENRES_Asian',
                'Avant-Garde': 'GENRES_AvantGarde',
            },
        }
        # Create the tokenizer and add to the vocabulary
        tokenizer = REMI()
        add_conditions_to_vocab(tokenizer, condition_mapping)
        # Create the metadata for a track
        metadata = {
            "pianist": "Billy Bob",
            "genres": [
                {
                    "name": "African",
                    "weight": 5
                },
                {
                    "name": "Avant-Garde",
                    "weight": 9
                }
            ]
        }
        expected = [tokenizer["GENRES_African"], tokenizer["GENRES_AvantGarde"], tokenizer["PIANIST_BillyBob"]]
        actual_strs = cond.get_conditions_for_track(condition_mapping, metadata, tokenizer)
        actual = [tokenizer[t] for t in actual_strs]
        self.assertEqual(expected, actual)

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

    def test_get_tempo_token(self):
        # Create the tokenizer and add to the vocabulary
        tokenizer = REMI()
        # Should raise an error if we haven't added the time signature tokens in yet
        with self.assertRaises(AssertionError):
            _ = cond.get_tempo_token(100, tokenizer)
        # Add in tempo tokens from [100, 110, 120, ..., 200]
        add_tempos_to_vocab(tokenizer, (100, 200), 11)
        # Test with 154 BPM
        expected = "TEMPOCUSTOM_150"
        actual = cond.get_tempo_token(154, tokenizer)
        self.assertEqual(expected, actual)
        # Test with 156 BPM
        expected = "TEMPOCUSTOM_160"
        actual = cond.get_tempo_token(156, tokenizer)
        self.assertEqual(expected, actual)
        # Raise an error if we try an out of range value
        with self.assertRaises(ValueError):
            _ = cond.get_tempo_token(20000, tokenizer)


if __name__ == '__main__':
    unittest.main()
