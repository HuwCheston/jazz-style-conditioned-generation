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

    def test_get_condition_values(self):
        condition = "genres"
        # Results will contain duplicates here, we de-dupe in get_mapping_for_condition
        expected = [
            "artist_genre1",
            "artist_genre2",
            "album_genre1",
            "album_genre2",
            "artist_genre1",
            "artist_genre3"
        ]
        actual = cond.get_condition_values(condition, TEST_DICT)
        self.assertEqual(actual, expected)

    def test_get_mapping_for_condition(self):
        condition = "genres"
        expected = {"album_genre1": 0, "album_genre2": 1, "artist_genre1": 2, "artist_genre2": 3, "artist_genre3": 4}
        actual = cond.get_mapping_for_condition(condition, TEST_DICT)
        self.assertEqual(actual, expected)

    def test_get_condition_special_tokens(self):
        # Testing for genres
        condition = "genres"
        mapping = {"album_genre1": 0, "album_genre2": 1, "artist_genre1": 2, "artist_genre2": 3, "artist_genre3": 4}
        expected = ["GENRES_0", "GENRES_1", "GENRES_2", "GENRES_3", "GENRES_4"]
        actual = cond.get_special_tokens_for_condition(condition, mapping)
        self.assertEqual(actual, expected)
        # Testing for artists
        condition = "pianist"
        mapping = {"Bill Evans": 0, "Abdullah Ibrahim": 1, "Billy Bob": 2, "Jimmy Jim": 3}
        expected = ["PIANIST_0", "PIANIST_1", "PIANIST_2", "PIANIST_3"]
        actual = cond.get_special_tokens_for_condition(condition, mapping)
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
