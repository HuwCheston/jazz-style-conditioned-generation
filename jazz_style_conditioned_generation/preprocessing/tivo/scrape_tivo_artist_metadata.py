#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Scrapes artist-level metadata for artists from JTD, PiJAMA, Pianist8"""

import os

from loguru import logger
from tqdm import tqdm

from jazz_style_conditioned_generation import utils
from jazz_style_conditioned_generation.preprocessing.tivo.tivo_utils import (
    API_ROOT,
    DATA_ROOT,
    format_named_person_or_entity,
    cached_api_call,
    clean_prose_text
)

API_ARTIST_SEARCH = f'{API_ROOT}/search/artist'
ARTIST_METADATA_FIELDS_WITH_WEIGHTS = ["moods", "musicGenres", "similars"]

# The first (most relevant) hit on TiVo for these is NOT the correct artist. This mapping gives the correct result
HIT_IDXS = {
    "John Taylor": 1,
    "Edward Simon": 1
}


def get_artists():
    """Gets the names of all artists contained within the datasets that have TiVo metadata (Pianist8, PiJAMA, JTD)"""
    # i.e., ./data/raw/jtd or ./data/raw/pijama
    pianists = []
    for dataset in utils.DATASETS_WITH_TIVO:
        dataset_dir = os.path.join(DATA_ROOT, dataset)
        # ./data/raw/jtd/<track>, ./data/raw/pijama/<track>
        for track in os.listdir(dataset_dir):
            track_dir = os.path.join(dataset_dir, track)
            # This will happen if we have e.g., .gitkeep files
            if not os.path.isdir(track_dir):
                continue
            # ./data/raw/jtd/<track>/metadata.json, ./data/raw/pijama/<track>/metadata.json, ...
            metadata_path = os.path.join(track_dir, 'metadata.json')
            # We need to raise errors as all tracks should have this metadata by default
            if not os.path.isfile(metadata_path):
                raise FileNotFoundError(f'Could not find metadata for track {track}, dataset {dataset}')
            # Load the metadata file and get the name of the pianist
            metadata_loaded = utils.read_json_cached(metadata_path)
            pianists.append(metadata_loaded["pianist"])
    # Remove duplicates and sort alphabetically
    return sorted(list(set(pianists)))


def artist_search(artist_name: str) -> dict:
    """Make an API call to get metadata for an artist"""
    artist_name = format_named_person_or_entity(artist_name).replace(' ', '+')
    request_fmt = f'{API_ARTIST_SEARCH}?name={artist_name}&includeAllFields=true&limit=5'
    return cached_api_call(request_fmt)


def parse_artist_bios(tivo_musicbio: dict) -> list:
    """TiVo artist bios have a weird structure, with numerous different keys. Here, we unpack them to a flat list"""
    all_bios = []
    # These are the keys that we find
    for bio_key1 in ['headlineBio', 'biography', 'musicBioOverviewEnglish']:
        if bio_key1 in tivo_musicbio.keys():
            # Sometimes, we'll get a list of dictionaries
            if isinstance(tivo_musicbio[bio_key1], list):
                for txt in tivo_musicbio[bio_key1]:
                    # The actual text can be stored under different keys, as well
                    for bio_key2 in ['text', 'overview']:
                        try:
                            all_bios.append(clean_prose_text(txt[bio_key2]))
                        except KeyError:
                            continue
            # other times, we'll just get a single dictionary
            else:
                all_bios.append(clean_prose_text(tivo_musicbio[bio_key1]))
    return all_bios


def parse_tivo_artist_metadata(artist_name: str, tivo_artist_lookup: dict) -> dict:
    """Parse metadata from the TiVo API for a given artist"""
    assert tivo_artist_lookup['hitCount'] > 0, 'Got no hits for an artist that should have them!'
    # We grab the index of the correct hit from the dictionary
    if artist_name in HIT_IDXS.keys():
        hit_idx = HIT_IDXS[artist_name]
    # Otherwise, we can use the first hit for the artist (these have been confirmed by hand)
    else:
        hit_idx = 0
    # Grab the corresponding hit
    hit = tivo_artist_lookup['hits'][hit_idx]
    new_artist_metadata = {}
    # These are the fields that have separate "id", "name", and "weight" key-value pairs
    for field in ARTIST_METADATA_FIELDS_WITH_WEIGHTS:
        # We don't always get every field for every album
        if field in hit.keys():
            # Add just the required fields into our list
            new_artist_metadata[field.replace("musicGenres", "genres")] = [
                {k: v for k, v in value.items() if k in ['name', 'weight']}
                for value in hit[field]
            ]
        else:
            # Otherwise, just set this field to an empty list
            new_artist_metadata[field.replace("musicGenres", "genres")] = []
    # Sanity check: store matched artist name
    new_artist_metadata['tivo_artist_name'] = hit['name']
    # Now, add all the artist biographies
    new_artist_metadata['bio'] = []
    if "musicBio" in hit.keys():
        new_artist_metadata['bio'].extend(parse_artist_bios(hit['musicBio']))
    if len(new_artist_metadata['bio']) == 0:
        logger.warning(f"No bio found for artist {artist_name}!")
    return new_artist_metadata


def parse_similar_artists(similar_artists: list[dict], dataset_artists: list[str]) -> list[dict]:
    """Keep artists in `similar_artists` that are also contained in `dataset_artists`"""
    low_artists = [i.lower() for i in dataset_artists]
    for similar in similar_artists:
        if similar["name"].lower() in low_artists:
            yield similar


def main():
    """Process metadata for all artists and dump in the references folder"""
    # Get the names of all unique artists
    all_artists = get_artists()
    logger.info(f"Found {len(all_artists)} unique artists")
    # Iterate over every artist
    for artist in tqdm(all_artists, desc="Processing..."):
        # Search for the artist and parse the metadat
        searched = artist_search(artist)
        parsed = parse_tivo_artist_metadata(artist, searched)
        # Update the similar artist list to only include other artists that are contained in the dataset
        parsed["similars"] = list(parse_similar_artists(parsed["similars"], all_artists))
        if len(parsed["similars"]) == 0:
            logger.warning(f"No similar artists found for {artist}!")
        # Dump to a JSON inside the reference directory
        artist_fname = utils.remove_punctuation(artist).replace(" ", "")
        json_fpath = os.path.join(utils.get_project_root(), 'references/tivo_artist_metadata', artist_fname + '.json')
        utils.write_json(parsed, json_fpath)
    logger.info('Done!')


if __name__ == "__main__":
    main()
