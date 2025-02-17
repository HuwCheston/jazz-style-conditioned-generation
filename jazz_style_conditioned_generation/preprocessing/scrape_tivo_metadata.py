#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Adds metadata from TiVo into the existing metadata for JTD + PiJAMA"""

import json
import os
import re
from functools import lru_cache

import requests
from loguru import logger
from thefuzz import fuzz
from tqdm import trange

from jazz_style_conditioned_generation import utils

DATA_ROOT = os.path.join(utils.get_project_root(), 'data/raw')
DATASETS = ["jtd", 'pijama']
OVERWRITE_EXISTING = True  # if True, will re-process all tracks; if False, will skip

# Matched album name + matched artist name must be over this value to be valid
MIN_ALBUM_VALIDATION_SCORE = 180  # maximum similarity is 200

API_ROOT = "https://tivomusicapi-staging-elb.digitalsmiths.net/sd/tivomusicapi/taps/v3"
API_HEADERS = {"Accept": "application/json"}
API_ALBUM_SEARCH = f'{API_ROOT}/search/album'
API_ALBUM_LOOKUP = f'{API_ROOT}/lookup/album'
API_ARTIST_SEARCH = f'{API_ROOT}/search/artist'
API_WAIT_TIME = 0.1  # seconds, API terms of use specify no more than five calls per second

ALBUM_METADATA_FIELDS = [
    "album_moods", "album_genres", "album_themes", "album_flags", "album_review", "tivo_album_artists"
]
ALBUM_METADATA_FIELDS_WITH_WEIGHTS = ["moods", "genres", "subGenres", "themes"]

ARTIST_METADATA_FIELDS = ["artist_moods", "artist_genres", "artist_bio"]
ARTIST_METADATA_FIELDS_WITH_WEIGHTS = ["moods", "musicGenres"]

ALBUM_HITS, ALBUM_MISSES = 0, []


def format_named_person_or_entity(npe: str):
    return " ".join(npe.lstrip().rstrip().title().split())


def add_missing_keys(di: dict, keys: list, value_type: type = list) -> dict:
    """Adds key-value pairs that do not exist into a dictionary"""
    for key in keys:
        if key not in di.keys():
            di[key] = value_type()  # defaults to an empty list
    return di


def get_tracks() -> list[str]:
    """Gets the names of all tracks contained within JTD and PiJAMA"""
    # i.e., ./data/raw/jtd or ./data/raw/pijama
    for dataset in DATASETS:
        dataset_dir = os.path.join(DATA_ROOT, dataset)
        # ./data/raw/jtd/<track>, ./data/raw/pijama/<track>
        for track in os.listdir(dataset_dir):
            track_dir = os.path.join(dataset_dir, track)
            # This will happen if we have e.g., .gitkeep files
            if not os.path.isdir(track_dir):
                continue
            # ./data/raw/jtd/<track>/metadata.json, ./data/raw/pijama/<track>/metadata.json
            metadata_path = os.path.join(track_dir, 'metadata.json')
            # We need to raise errors as all tracks should have this metadata by default
            if not os.path.isfile(metadata_path):
                raise FileNotFoundError(f'Could not find metadata for track {track}, dataset {dataset}')
            yield track_dir


@utils.wait(secs=API_WAIT_TIME)
@lru_cache(maxsize=None)
def _cached_api_call(url: str) -> dict:
    """Makes an API call to a given url: waiting and caching are implemented to prevent rate limiting"""
    return requests.get(url, headers=API_HEADERS).json()


def album_search(track_metadata: dict) -> str:
    """Makes an API request to search for an album given an artist and title"""
    artist = format_named_person_or_entity(track_metadata['bandleader']).replace(' ', '+')
    album = format_named_person_or_entity(track_metadata['album_name']).replace(' ', "+")
    request_fmt = f"{API_ALBUM_SEARCH}?artistName={artist}&title={album}"
    return _cached_api_call(request_fmt)


def validate_album(track_metadata: dict, track_hit: dict) -> bool:
    """We can get multiple matches for a single album, but we only want to get the most relevant one"""
    # Mark this album as invalid if we don't have any matches for it
    if track_hit['hitCount'] == 0:
        return False
    matches = []
    # Every album is treated as a "hit"
    for hit in track_hit['hits']:
        # Get the name of the album and compute similarity to the expected name
        expected_album_name = format_named_person_or_entity(track_metadata['album_name'])
        title_match = fuzz.partial_ratio(expected_album_name, hit['title'])
        # Iterate over all artists associated with the album
        artist_match = 0
        expected_artist_name = format_named_person_or_entity(track_metadata['bandleader'])
        if 'primaryArtists' in hit.keys():
            for artist in hit['primaryArtists']:
                # Get the maximum similarity with the expected artist
                this_artist_match = fuzz.ratio(expected_artist_name, artist['name'])
                artist_match = max(this_artist_match, artist_match)
        # Sum both string similarities together --- artist + album similarity
        matches.append(sum([artist_match, title_match]))
    # Get the closest matching result
    best_match = max(matches)
    # If this is above our threshold, we can use the album
    if best_match >= MIN_ALBUM_VALIDATION_SCORE:
        return track_hit['hits'][matches.index(best_match)]['id']
    # Otherwise, mark it as invalid
    else:
        return False


def album_lookup(tivo_album_id: str) -> dict:
    """Makes an API request using the unique ID for an album to get complete metadata"""
    request_fmt = f'{API_ALBUM_LOOKUP}?albumId={tivo_album_id}'
    return _cached_api_call(request_fmt)


def clean_prose_text(prose_text: str) -> str:
    """Prose text from TiVo (e.g., bios, reviews) contains some HTML tags which we need to remove"""
    # Iterate over each markup tag
    for remove in ["roviLink", "muzeItalic"]:
        # Use some regex to remove the tags but keep the content between the tags
        prose_text = re.sub(rf'\[{remove}.*?\](.*?)\[/{remove}\]', r'\1', prose_text)
    return prose_text


def parse_tivo_album_metadata(tivo_album_lookup: dict):
    """Parse metadata from the TiVo API for a given album"""
    assert tivo_album_lookup['hitCount'] > 0, 'Got no hits for an album that should have them!'
    # We're searching by album ID, so there should only ever be one match
    hit = tivo_album_lookup['hits'][0]
    new_album_metadata = {}
    # These are the fields that have separate "id", "name", and "weight" key-value pairs
    for field in ALBUM_METADATA_FIELDS_WITH_WEIGHTS:
        # We'll use this to combine sub-genres and genres into a single list
        field_replace = field.replace('subGenres', 'genres')
        if f'album_{field_replace}' not in new_album_metadata.keys():
            new_album_metadata[f'album_{field_replace}'] = []
        if field in hit.keys():
            new_album_metadata[f'album_{field_replace}'].extend([
                {k: v for k, v in value.items() if k in ['name', 'weight']}
                for value in hit[field]
            ])
    # Sanity check: store matched artist and title
    new_album_metadata['tivo_album_name'] = hit['title']
    new_album_metadata['tivo_album_artists'] = [i['name'] for i in hit['primaryArtists']]
    # Extracting reviews for the album
    if "primaryReview" in hit.keys():
        new_album_metadata['album_review'] = [clean_prose_text(review['text']) for review in hit['primaryReview']]
    else:
        new_album_metadata['album_review'] = []
    # Extracting flags for the album
    if "flags" in hit.keys():
        new_album_metadata['album_flags'] = hit['flags']
    else:
        new_album_metadata['album_flags'] = []
    return new_album_metadata


def artist_search(artist_name: str) -> dict:
    """Make an API call to get metadata for an artist"""
    artist_name = format_named_person_or_entity(artist_name).replace(' ', '+')
    request_fmt = f'{API_ARTIST_SEARCH}?name={artist_name}&includeAllFields=true&limit=1'
    return _cached_api_call(request_fmt)


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


def parse_tivo_artist_metadata(tivo_artist_lookup: dict) -> dict:
    """Parse metadata from the TiVo API for a given artist"""
    assert tivo_artist_lookup['hitCount'] > 0, 'Got no hits for an artist that should have them!'
    # We can fairly safely use the first hit for the artist, all the names are pretty unique
    hit = tivo_artist_lookup['hits'][0]
    new_artist_metadata = {}
    # These are the fields that have separate "id", "name", and "weight" key-value pairs
    for field in ARTIST_METADATA_FIELDS_WITH_WEIGHTS:
        # We don't always get every field for every album
        if field in hit.keys():
            # Add just the required fields into our list
            new_artist_metadata[f'artist_{field.replace("musicGenres", "genres")}'] = [
                {k: v for k, v in value.items() if k in ['name', 'weight']}
                for value in hit[field]
            ]
        else:
            # Otherwise, just set this field to an empty list
            new_artist_metadata[f'artist_{field.replace("musicGenres", "genres")}'] = []
    # Sanity check: store matched artist name
    new_artist_metadata['tivo_artist_name'] = hit['name']
    # Now, add all the artist biographies
    new_artist_metadata['artist_bio'] = []
    if "musicBio" in hit.keys():
        new_artist_metadata['artist_bio'].extend(parse_artist_bios(hit['musicBio']))
    return new_artist_metadata


def dump_metadata_json(metadata_dict: dict, filepath: str) -> None:
    """Dumps a dictionary as a JSON in provided location"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(metadata_dict, f, indent=4, ensure_ascii=False, sort_keys=False)


def parse_all_metadata(track_path: str) -> dict:
    """For a given track, get track + album + artist metadata from TiVo and return a single dictionary"""
    global ALBUM_HITS, ALBUM_MISSES

    # Get the metadata for the track (already available for JTD + PiJAMA)
    track_metadata = utils.read_json_cached(os.path.join(track_path, 'metadata.json'))
    # Parse metadata for the ARTIST: this is always the pianist
    artist_metadata = artist_search(track_metadata['pianist'])
    artist_metadata_parsed = parse_tivo_artist_metadata(artist_metadata)
    # Parse metadata for the ALBUM: this may be lead by a different musician to the pianist!
    album_searched = album_search(track_metadata)
    album_is_valid = validate_album(track_metadata, album_searched)
    # If we've been able to find a matching album
    if album_is_valid:
        ALBUM_HITS += 1
        # Make another API call to get the full set of metadata and parse it
        album_metadata = album_lookup(album_is_valid)
        album_metadata_parsed = parse_tivo_album_metadata(album_metadata)
    else:
        track_sep = track_path.split(os.path.sep)[-1]
        ALBUM_MISSES.append(track_sep)
        logger.warning(f'Miss! {track_sep}')
        album_metadata_parsed = {}
    # Combine all the metadata dictionaries into a single dictionary
    all_metadata = (
            track_metadata |
            add_missing_keys(artist_metadata_parsed, ARTIST_METADATA_FIELDS) |
            add_missing_keys(album_metadata_parsed, ALBUM_METADATA_FIELDS)
    )
    return add_missing_keys(all_metadata, ["tivo_artist_name", "tivo_album_name"], str)


def main():
    global ALBUM_HITS, ALBUM_MISSES

    # Get paths to all tracks in both JTD + PiJAMA
    all_tracks = sorted(list(get_tracks()))  # sorting ensures we always process in the same order
    logger.info(f'Found {len(all_tracks)} tracks to get TiVo metadata for!')
    with trange(len(all_tracks), desc='Processing...') as t:
        # Iterate over every track in both datasets
        for track_path in all_tracks:
            # This is where we'll save our new metadata
            tivo_metadata_path = os.path.join(track_path, 'metadata_tivo.json')
            # If we've already processed the track, we can skip over it
            if os.path.isfile(tivo_metadata_path) and not OVERWRITE_EXISTING:
                logger.info(f'Track {track_path} already processed, skipping...')
                ALBUM_HITS += 1
            # Otherwise, grab the metadata from the API and dump in the folder for this track
            else:
                track_metadata = parse_all_metadata(track_path)
                dump_metadata_json(track_metadata, tivo_metadata_path)
            # Update the TQDM progress bar
            t.set_postfix(hits=ALBUM_HITS, misses=len(ALBUM_MISSES))
            t.update(1)
    # List the tracks that are misses in the console
    misses_fmt = "\n".join(ALBUM_MISSES)
    logger.info('Done!')
    logger.warning(rf'These tracks were misses: {misses_fmt}')


if __name__ == "__main__":
    assert os.path.isdir(DATA_ROOT), f'Could not find data at {DATA_ROOT}!'
    main()
