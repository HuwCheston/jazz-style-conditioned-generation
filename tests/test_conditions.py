#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for creating conditions"""

import unittest

from jazz_style_conditioned_generation.data import conditions as cond

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


if __name__ == '__main__':
    unittest.main()
