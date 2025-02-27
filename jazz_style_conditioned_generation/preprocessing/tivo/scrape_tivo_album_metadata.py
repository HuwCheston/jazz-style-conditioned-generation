#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Adds metadata from TiVo into the existing album metadata for JTD, PiJAMA, and Pianist8"""

import os

from loguru import logger
from thefuzz import fuzz
from tqdm import trange

from jazz_style_conditioned_generation import utils
from jazz_style_conditioned_generation.preprocessing.tivo.tivo_utils import (
    API_ROOT,
    TIVO_DATASETS,
    DATA_ROOT,
    format_named_person_or_entity,
    cached_api_call,
    clean_prose_text,
    add_missing_keys
)

OVERWRITE_EXISTING = True  # if True, will re-process all tracks; if False, will skip

# Matched album name + matched artist name must be over this value to be valid
MIN_ALBUM_VALIDATION_SCORE = 180  # maximum similarity is 200

API_ALBUM_SEARCH = f'{API_ROOT}/search/album'
API_ALBUM_LOOKUP = f'{API_ROOT}/lookup/album'

ALBUM_METADATA_FIELDS = [
    "album_moods", "album_genres", "album_themes", "album_flags", "album_review", "tivo_album_artists"
]
ALBUM_METADATA_FIELDS_WITH_WEIGHTS = ["moods", "genres", "subGenres", "themes"]

ALBUM_HITS, ALBUM_MISSES = 0, []


def get_tracks() -> list[str]:
    """Gets the names of all tracks contained within the datasets that have TiVo metadata (Pianist8, PiJAMA, JTD)"""
    # i.e., ./data/raw/jtd or ./data/raw/pijama
    for dataset in TIVO_DATASETS:
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


def album_search(track_metadata: dict) -> str:
    """Makes an API request to search for an album given an artist and title"""
    artist = format_named_person_or_entity(track_metadata['bandleader']).replace(' ', '+')
    album = format_named_person_or_entity(track_metadata['album_name']).replace(' ', "+")
    request_fmt = f"{API_ALBUM_SEARCH}?artistName={artist}&title={album}"
    return cached_api_call(request_fmt)


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
    return cached_api_call(request_fmt)


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


def parse_all_metadata(track_path: str) -> dict:
    """For a given track, get track + album + artist metadata from TiVo and return a single dictionary"""
    global ALBUM_HITS, ALBUM_MISSES

    # Get the metadata for the track (already available for JTD + PiJAMA)
    track_metadata = utils.read_json_cached(os.path.join(track_path, 'metadata.json'))
    # Parse metadata for the ALBUM: this may be lead by a different musician to the pianist!
    album_searched = album_search(track_metadata)
    album_is_valid = validate_album(track_metadata, album_searched)
    # If we've been able to find a matching album
    if album_is_valid:
        ALBUM_HITS += 1
        # Make another API call to get the full set of metadata and parse it
        album_metadata = album_lookup(album_is_valid)
        album_metadata_parsed = parse_tivo_album_metadata(album_metadata)
        album_metadata_parsed["tivo_found_match"] = True
    else:
        track_sep = track_path.split(os.path.sep)[-1]
        ALBUM_MISSES.append(track_sep)
        logger.warning(f'Miss! {track_sep}')
        album_metadata_parsed = {"tivo_found_match": False}
    # Combine all the metadata dictionaries into a single dictionary
    all_metadata = (
            track_metadata |
            # add_missing_keys(artist_metadata_parsed, ARTIST_METADATA_FIELDS) |
            add_missing_keys(album_metadata_parsed, ALBUM_METADATA_FIELDS)
    )
    return add_missing_keys(all_metadata, ["tivo_album_name"], str)


def main():
    global ALBUM_HITS, ALBUM_MISSES

    # Get paths to all tracks in both JTD + PiJAMA
    all_tracks = sorted(list(get_tracks()))  # sorting ensures we always process in the same order
    logger.info(f'Found {len(all_tracks)} tracks to get TiVo metadata for!')
    with trange(len(all_tracks), desc='Processing...') as t:
        # Iterate over every track in both datasets
        for track_path in all_tracks[50:]:
            # This is where we'll save our new metadata
            tivo_metadata_path = os.path.join(track_path, 'metadata_tivo.json')
            # If we've already processed the track, we can skip over it
            if os.path.isfile(tivo_metadata_path) and not OVERWRITE_EXISTING:
                logger.info(f'Track {track_path} already processed, skipping...')
                ALBUM_HITS += 1
            # Otherwise, grab the metadata from the API and dump in the folder for this track
            else:
                track_metadata = parse_all_metadata(track_path)
                utils.write_json(track_metadata, tivo_metadata_path)
            # Update the TQDM progress bar
            t.set_postfix(hits=ALBUM_HITS, misses=len(ALBUM_MISSES))
            t.update(1)
    # TODO: we should create an empty metadata_tivo file for tracks from bushgrafts/jja?
    # List the tracks that are misses in the console
    misses_fmt = "\n".join(ALBUM_MISSES)
    logger.info('Done!')
    logger.warning(rf'These tracks were misses: {misses_fmt}')


if __name__ == "__main__":
    assert os.path.isdir(DATA_ROOT), f'Could not find data at {DATA_ROOT}!'
    utils.seed_everything(utils.SEED)
    main()
