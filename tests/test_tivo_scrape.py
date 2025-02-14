#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for scraping metadata for tracks from TiVo"""

import unittest

from deepdiff import DeepDiff

from jazz_style_conditioned_generation.preprocessing import scrape_tivo_metadata as stm


def compare_nested_dicts(d1, d2) -> int:
    """Checks if two nested dictionaries are equal, ignoring order"""
    return len(list(DeepDiff(d1, d2, ignore_order=True).keys())) == 0


class TivoScrape(unittest.TestCase):
    def test_format_name(self):
        # Test stripping white space
        tester = " Bill Evans "
        expected = "Bill Evans"
        actual = stm.format_named_person_or_entity(tester)
        self.assertEqual(expected, actual)
        # Test cases
        tester = "album name  "
        expected = "Album Name"
        actual = stm.format_named_person_or_entity(tester)
        self.assertEqual(expected, actual)

    def test_add_missing_keys(self):
        # Test with adding empty list
        tester = {"artist": "Bill Evans"}
        expected = {"artist": "Bill Evans", "tmp": []}
        actual = stm.add_missing_keys(tester, ["tmp"], list)
        self.assertEqual(expected, actual)
        # Test with adding empty string
        tester = {"artist": "Bill Evans"}
        expected = {"artist": "Bill Evans", "tmp": ""}
        actual = stm.add_missing_keys(tester, ["tmp"], str)
        self.assertEqual(expected, actual)

    def test_validate_album(self):
        # Test an album without any hits
        bad_album = {"hitCount": 0}
        actual = stm.validate_album({}, bad_album)
        self.assertFalse(actual)
        # Test an album with valid hits
        good_album = {
            "hitCount": 1,
            "hits": [
                {
                    "id": "good-good",
                    "title": "Albumy Album",
                    "primaryArtists": [
                        {
                            "name": "Artisty Artist"
                        }
                    ]
                }
            ]
        }
        good_metadata = {
            "bandleader": "Artisty Artist",
            "album_name": "Albumy Album",
        }
        expected = "good-good"
        actual = stm.validate_album(good_metadata, good_album)
        self.assertEqual(expected, actual)
        # Test an album without valid hits
        bad_metadata = {
            "bandleader": "12334oiwejf",
            "album_name": "1239u34091"
        }
        self.assertFalse(stm.validate_album(bad_metadata, good_album))

    # def test_clean_text(self):
    #     tester = "I am some text with [roviLink]markup[\roviLink] that [muzeItalic]should[\muzeItalic] get removed"
    #     expected = "I am some text with markup that should get removed"
    #     actual = stm.clean_prose_text(tester)
    #     self.assertEqual(expected, actual)

    def test_parse_album(self):
        # Test an album without any hits
        bad_album = {"hitCount": 0}
        self.assertRaises(AssertionError, stm.parse_tivo_album_metadata, bad_album)
        good_album = {
            "hitCount": 1,
            "hits": [
                {
                    "moods": [
                        {
                            "name": "moody-mood",
                            "id": "bye-bye",
                            "weight": 12
                        }
                    ],
                    "themes": [
                        {
                            "name": "themey-theme",
                            "id": "bye-bye",
                            "weight": 1
                        }
                    ],
                    "subGenres": [
                        {
                            "name": "subgenrey-subgenre",
                            "id": "bye-bye",
                            "weight": 4
                        },
                        {
                            "name": "subgenrey2-subgenre2",
                            "id": "bye-bye",
                            "weight": 1
                        },
                    ],
                    "genres": [
                        {
                            "name": "genre-genre",
                            "id": "bye-bye",
                            "weight": 123
                        }
                    ],
                    "primaryReview": [{"text": "text text"}],
                    "flags": ["flag1", "flag2"]
                }
            ]
        }
        expected = {
            "album_moods": [
                {
                    "name": "moody-mood",
                    "weight": 12
                }
            ],
            "album_themes": [
                {
                    "name": "themey-theme",
                    "weight": 1
                }
            ],
            "album_genres": [
                {
                    "name": "subgenrey-subgenre",
                    "weight": 4
                },
                {
                    "name": "subgenrey2-subgenre2",
                    "weight": 1
                },
                {
                    "name": "genre-genre",
                    "weight": 123
                },
            ],
            "album_review": ["text text"],
            "album_flags": ["flag1", "flag2"],
        }
        actual = stm.parse_tivo_album_metadata(good_album)
        self.assertTrue(compare_nested_dicts(expected, actual))

    def test_parse_artist_bio(self):
        pass

    def test_parse_artist(self):
        pass


if __name__ == '__main__':
    unittest.main()
