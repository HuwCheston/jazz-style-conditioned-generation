#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for dataloader"""

import unittest

import torch

from jazz_style_conditioned_generation.data.dataloader import CollatorMIDICondition, DatasetMIDICondition


class DataloaderTest(unittest.TestCase):
    DS = DatasetMIDICondition(
        condition_mapping={
            "genres": {
                "Acoustic Jazz": 0,
                "African Folk": 1,
                "Math Rock": 2,
            },
            "pianist": {
                "Jimmy Jim": 0,
                "Bobby Bob": 1,
                "Freddie Fred": 2
            }
        },
        combine_artist_and_album_tags=False,
        files_paths=["a/fake/filepath"],
        tokenizer=None,
        max_seq_len=1024,
        bos_token_id=1,
        eos_token_id=2,
    )

    def test_get_conditions_from_metadata(self):
        dummy_metadata = {
            "album_genres": [
                {"name": "Math Rock"},
                {"name": "African Folk"},
            ],
            "artist_genres": [
                {"name": "Acoustic Jazz"}
            ],
            "album_themes": [
                {"name": "Themey Theme"}
            ],
            "pianist": "Freddie Fred"
        }
        # First, we just want to gather the album-level genres
        expected = {"genres": ["Math Rock", "African Folk"], "pianist": ["Freddie Fred"]}
        actual = self.DS.get_conditions(dummy_metadata)
        self.assertEqual(expected, actual)
        # Now, we want to gather both the artist-level genre tags and the album-level genre tags
        self.DS.combine_artist_and_album_tags = True
        expected = {"genres": ["Math Rock", "African Folk", "Acoustic Jazz"], "pianist": ["Freddie Fred"]}
        actual = self.DS.get_conditions(dummy_metadata)
        self.assertEqual(expected, actual)
        # Set back to default
        self.DS.combine_artist_and_album_tags = False

    def test_conditions_to_tokens(self):
        dummy_results = {"genres": ["Math Rock", "African Folk"], "pianist": ["Freddie Fred"]}
        expected = ["GENRES_2", "GENRES_1", "PIANIST_2"]
        actual = self.DS.conditions_to_tokens(dummy_results)
        self.assertEqual(expected, actual)

    def test_ensemble_token(self):
        # Test with JTD
        fname = "a/fake/filepath/jtd/track/piano_midi.mid"
        expected = "ENSEMBLE_0"
        actual = self.DS.get_ensemble_context_token(fname)
        self.assertEqual(expected, actual)
        # Test with PiJama
        fname = "a/fake/filepath/pijama/track/piano_midi.mid"
        expected = "ENSEMBLE_1"
        actual = self.DS.get_ensemble_context_token(fname)
        self.assertEqual(expected, actual)
        # Should raise an error
        bad_fname = "a/fake/filepath/with/no/dataset/name"
        self.assertRaises(ValueError, self.DS.get_ensemble_context_token, bad_fname)

    def test_add_special_tokens(self):
        input_ids = torch.tensor([1, 2, 3])
        special_tokens = torch.tensor([5, 5, 5])
        # By default, we add at position 1, which is the "beginning of sequence" tag
        expected = [1, 5, 5, 5, 2, 3]
        actual = self.DS.add_special_tokens_to_input(input_ids, special_tokens).tolist()
        self.assertEqual(expected, actual)


class CollatorMIDIConditionTest(unittest.TestCase):
    DUMMY_BATCH = [
        (
            dict(
                input_ids=torch.randint(0, 100, (1000,)),
                labels=torch.randint(0, 10, (20,))
            ),
            dict(
                condition={
                    "genres": ["genre_1", "genre_2"],
                    "moods": ["mood_1", "mood_2"]
                },
                condition_tokens=[
                    "GENRES_0", "GENRES_1", "MOODS_0", "MOODS_1"
                ]
            ),
        ),
        (
            dict(
                input_ids=torch.randint(0, 100, (500,)),  # will be padded
                labels=torch.randint(0, 10, (10,))
            ),
            dict(
                condition={
                    "genres": ["genre_2", "genre_3"],
                    "moods": ["mood_1", "mood_2", "mood_3"]
                },
                condition_tokens=[
                    "GENRES_1", "GENRES_2", "MOODS_0", "MOODS_1", "MOODS_2"
                ]
            )
        )
    ]
    CL = CollatorMIDICondition(
        pad_token_id=0,
        labels_pad_idx=0,
        copy_inputs_as_labels=False,
        pad_on_left=False  # This will probably be True for the models we actually train...
    )
    OUT = CL(DUMMY_BATCH)

    def test_keys(self):
        # Test we have all the required keys
        expected_keys = ["input_ids", "labels", "attention_mask", "condition", "condition_tokens"]
        actual_keys = list(self.OUT.keys())
        self.assertEqual(expected_keys, actual_keys)

    def test_conditions(self):
        # Test we have maintained the same order of conditions
        actual_conditions = self.OUT["condition"][0]
        expected_conditions = {
            "genres": ["genre_1", "genre_2"],
            "moods": ["mood_1", "mood_2"]
        }
        self.assertEqual(expected_conditions, actual_conditions)
        # Test that we've got the correct condition tokens
        actual_tokens = self.OUT["condition_tokens"][0]
        expected_tokens = ["GENRES_0", "GENRES_1", "MOODS_0", "MOODS_1"]
        self.assertEqual(expected_tokens, actual_tokens)

    def test_padding(self):
        # Test that we've padded the shorter labels correctly
        short_labels = self.OUT["labels"][1]
        self.assertEqual(short_labels.size(0), 20)
        zero_padded = short_labels[10:]
        self.assertTrue(all(i.item() == 0 for i in zero_padded))
        # Test that we've padded the input_ids
        short_ids = self.OUT["input_ids"][1]
        self.assertEqual(short_ids.size(0), 1000)
        zero_padded = short_labels[500:]
        self.assertTrue(all(i.item() == 0 for i in zero_padded))
        # TODO: test that the last item before padding is an end of sequence tag?

    def test_attention_mask(self):
        # Test that our attention mask corresponds with the input_ids
        for i, am in zip(self.OUT["input_ids"][1], self.OUT["attention_mask"][1]):
            i, am = i.item(), am.item()
            # i.e., not a pad token, attention should be calculated
            if i > 0:
                self.assertTrue(am == 1)
            # i.e., this is a pad token, attention should not be calculated
            else:
                self.assertTrue(i == am == 0)


if __name__ == '__main__':
    unittest.main()
