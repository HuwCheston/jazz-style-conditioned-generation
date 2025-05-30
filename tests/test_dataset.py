#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite to make sure everything is in order with our dataset. This will not be run on GitHub actions!"""

import os
import unittest

from miditok.constants import SCORE_LOADING_EXCEPTION
from symusic import Score

from jazz_style_conditioned_generation import utils

DATA_ROOT = os.path.join(utils.get_project_root(), "data/raw")
DATASETS = ["bushgrafts", "jja", "jtd", "pianist8", "pijama"]
ATEPP_ROOT = os.path.join(utils.get_project_root(), "data/pretraining/atepp")


@unittest.skipIf(os.getenv("REMOTE") == "true", "Skipping test on GitHub Actions")
class DatasetTester(unittest.TestCase):
    def test_individual_dataset_folders(self):
        """Check that we have individual folders for each of our datasets"""
        folder_ls = os.listdir(DATA_ROOT)
        for dataset in DATASETS:
            self.assertTrue(dataset in folder_ls)

    def test_track_numbers(self):
        """Test that we have the reported number of tracks for each dataset"""
        expected = {
            "pijama": 2777,  # as reported in paper
            "jtd": 1294,  # as reported in paper
            "jja": 5 * 9,  # five pianists, nine performances each
            "pianist8": 47,  # dataset contains 50 performances, but 3 are also in pijama
            "bushgrafts": 299  # total number we expect to scrape
        }
        for dataset, expected_tracks in expected.items():
            dataset_dir = os.path.join(DATA_ROOT, dataset)
            track_dirs = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]
            self.assertTrue(len(track_dirs) == expected_tracks)

    def test_tracks_have_required_files(self):
        """Test that we have all the required files for each track"""
        expected_files = ["piano_midi.mid", "metadata.json", "metadata_tivo.json"]
        for dataset in DATASETS:
            dataset_path = os.path.join(DATA_ROOT, dataset)
            for track in os.listdir(dataset_path):
                track_path = os.path.join(dataset_path, track)
                if not os.path.isdir(track_path):
                    continue
                all_files = os.listdir(track_path)
                for expected_file in expected_files:
                    self.assertTrue(expected_file in all_files)

    def test_metadata_has_required_fields(self):
        """Test that we have the expected fields in our metadata for tracks from different datasets"""
        expected_fields = {
            "track_name": DATASETS,
            "pianist": DATASETS,
            "tempo": ["jja", "bushgrafts", "jtd"],  # no tempo/time signature in PiJAMA/Pianist8
            "time_signature": ["jja", "bushgrafts", "jtd"],
            "genres": DATASETS,
            "moods": DATASETS,
            "themes": DATASETS
        }
        for dataset in DATASETS:
            dataset_path = os.path.join(DATA_ROOT, dataset)
            for track in os.listdir(dataset_path):
                track_path = os.path.join(dataset_path, track)
                if not os.path.isdir(track_path):
                    continue
                metadata_path = os.path.join(track_path, "metadata_tivo.json")
                metadata_loaded = utils.read_json_cached(metadata_path)
                for expected_field, dataset_names in expected_fields.items():
                    if dataset in dataset_names:
                        self.assertTrue(expected_field in metadata_loaded)
                    else:
                        self.assertFalse(expected_field in metadata_loaded)

    def test_artist_metadata(self):
        """Test that we have artist-level metadata inside references/tivo_artist_metadata"""
        # Get all of our artist-level metadata
        artist_metadatas = os.listdir(os.path.join(utils.get_project_root(), "references/tivo_artist_metadata"))
        self.assertTrue(len(artist_metadatas) == 129)
        # We expect to have artist-level metadata for these pianists
        for dataset in ["jtd", "pijama", "pianist8"]:
            metadatas = utils.get_data_files_with_ext(os.path.join(DATA_ROOT, dataset), "**/*_tivo.json")
            read = [utils.read_json_cached(js) for js in metadatas]
            for js in read:
                pianist_fmt = js["pianist"].replace(" ", "") + ".json"
                self.assertTrue(pianist_fmt in artist_metadatas)
        # We don't expect to have artist-level metadata for these pianists
        for dataset in ["jja", "bushgrafts"]:
            metadatas = utils.get_data_files_with_ext(os.path.join(DATA_ROOT, dataset), "**/*_tivo.json")
            read = [utils.read_json_cached(js) for js in metadatas]
            for js in read:
                pianist_fmt = js["pianist"].replace(" ", "") + ".json"
                self.assertFalse(pianist_fmt in artist_metadatas)

    def test_artist_metadata_fields(self):
        """Check that our artist-level metadata files have all the required fields"""
        expected_fields = ["moods", "similars", "genres"]
        artist_metadatas = os.path.join(utils.get_project_root(), "references/tivo_artist_metadata")
        for artist in os.listdir(artist_metadatas):
            artist_path = os.path.join(artist_metadatas, artist)
            artist_loaded = utils.read_json_cached(artist_path)
            for field in expected_fields:
                self.assertTrue(field in artist_loaded)

    def test_scores_load_ok(self):
        """Check that we can load all of our MIDI files without raising an exception"""
        midi_files = utils.get_data_files_with_ext(DATA_ROOT, "**/*.mid")
        for midi_file in midi_files:
            try:
                _ = Score(midi_file)
            except SCORE_LOADING_EXCEPTION as e:
                self.fail(f'{midi_file} raised error {e} when loading')

    def test_atepp_dataset(self):
        """Test the ATEPP dataset of classical piano transcriptions we use for pretraining"""
        for t in os.listdir(ATEPP_ROOT):
            # Check MIDI and metadata paths
            midi_fp = os.path.join(ATEPP_ROOT, t, "piano_midi.mid")
            self.assertTrue(os.path.exists(midi_fp))
            meta_fp = os.path.join(ATEPP_ROOT, t, "metadata_tivo.json")
            self.assertTrue(os.path.exists(meta_fp))
            # Check the MIDI file properties
            try:
                sc = Score(midi_fp)
            except SCORE_LOADING_EXCEPTION as e:
                self.fail(f"ATEPP file {midi_fp} raised error {e} when loading")
            else:
                # Check we only have one tempo, at 120 BPM
                self.assertTrue(len(sc.tempos) == 1)
                self.assertTrue(sc.tempos[0].qpm == utils.TEMPO)
                # Check we only have one time signature, at 4/4
                self.assertTrue(len(sc.time_signatures) == 1)
                self.assertTrue(sc.time_signatures[0].numerator == 4)
                # Check we only have one track
                self.assertTrue(len(sc.tracks) == 1)
                # Check the notes are all within the ranges of the piano keyboard
                min_pitch, max_pitch = utils.get_pitch_range(sc)
                self.assertTrue(min_pitch >= utils.MIDI_OFFSET)
                self.assertTrue(max_pitch <= utils.MIDI_OFFSET + utils.PIANO_KEYS)

    def test_tracks_in_split_exist(self):
        # Gather all tracks in all data splits
        all_tracks = []
        for split in ["train", "test", "validation"]:
            split_path = os.path.join(utils.get_project_root(), "references/data_splits", split + "_split.txt")
            with open(split_path, "r") as fin:
                tracks = fin.read().split("\n")
            all_tracks.extend([t for t in tracks if t != ""])
        # Go from split -> track
        for track in all_tracks:
            track_path = os.path.join(DATA_ROOT, track)
            self.assertTrue(os.path.exists(track_path))
        # Go from track -> split
        for data in DATASETS:
            data_path = os.path.join(DATA_ROOT, data)
            for track in os.listdir(data_path):
                # Skip over any .gitkeep files
                if track == ".gitkeep":
                    continue
                track = os.path.join(data, track)
                self.assertTrue(track in all_tracks)


if __name__ == '__main__':
    unittest.main()
