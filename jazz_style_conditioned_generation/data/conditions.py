#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Creates mappings for all conditions (performer, subgenre, mood, etc.)"""

from jazz_style_conditioned_generation import utils

# These are the only conditions we'll accept values for
# TODO: we can condition on context as well, i.e. PiJAMA or JTD
ACCEPT_CONDITIONS = ["moods", "genres", "pianist", "themes", "ensemble"]
# Each list should be populated with values for a condition that we don't want to use
EXCLUDE = {
    "genres": ["Jazz Instrument", "Jazz", "Piano Jazz"],  # i.e., nearly every track tagged with these values
    "moods": [],
    "pianist": [],
    "themes": [],
    "ensemble": []
}


def validate_conditions(conditions: list[str] | str) -> None:
    """Validates a single condition or list of conditions, raises ValuError for invalid conditions"""
    # Allows for a single string as input
    if isinstance(conditions, str):
        conditions = [conditions]
    for condition in conditions:
        if condition.lower() not in ACCEPT_CONDITIONS:
            raise ValueError(f'expected `condition` to be in {", ".join(ACCEPT_CONDITIONS)} but got {condition}')


def load_metadata_jsons(metadata_filepaths: list[str] = None) -> list[dict]:
    """Load all metadata JSONs corresponding to filepaths. Will use all tracks in ./data/root if no filepaths given"""
    if metadata_filepaths is None:
        metadata_filepaths = [j for j in utils.get_data_files_with_ext(ext='**/*.json') if j.endswith('_tivo.json')]
    for file in metadata_filepaths:
        yield utils.read_json_cached(file)


def get_inner_json_values(metadata: dict, key: str):
    """Get values from possibly-nested dictionary according to given key"""
    condition_val = metadata[key]
    # Genre, mood, and themes are all lists of dictionaries
    res = []
    if isinstance(condition_val, list):
        for condition_val_val in condition_val:
            res.append(condition_val_val['name'])  # we also have weight values here, but we're not getting them?
    # Performer is just a single string value
    else:
        res.append(condition_val)
    yield from res


def get_condition_values(condition: str, metadata_dicts: list[dict] | dict):
    """Given a condition in ACCEPT_CONDITIONS, extract all observed values from all metadata dicts"""
    # Check that the condition we've passed in is valid
    validate_conditions(condition)
    # Allow a single dictionary as input
    if isinstance(metadata_dicts, dict):
        metadata_dicts = [metadata_dicts]
    # We'll skip over these values
    res = []
    for metadata in metadata_dicts:
        # This will give us e.g., ["artist_genres", "album_genres"] for the input "genres"
        accept_keys = [k for k in metadata.keys() if condition.lower() in k.lower()]
        # Iterate over all of these keys and get the required value from the metadata dictionary
        for accept_key in accept_keys:
            res.extend(get_inner_json_values(metadata, accept_key))
    # Remove any keys for this condition that we don't want to use
    # i.e., [African Jazz, Bebop, Post-Bop] for genre, [Sophisticated, Relaxed, Frantic] for mood
    return [r for r in res if r not in EXCLUDE[condition]]


def get_mapping_for_condition(
        condition: str,
        metadata_dicts: list[dict] = None,
):
    """Gets key: idx mapping for all unique keys under a given condition"""
    # We can't get this information from the metadata
    if condition.lower() == "ensemble":
        return {"Trio": 0, "Solo": 1}
    if metadata_dicts is None:
        metadata_dicts = load_metadata_jsons(None)
    # Get the "raw" values for this condition (i.e., a list of all genres for all tracks, with duplicates)
    condition_values = get_condition_values(condition, metadata_dicts)
    # Remove duplicates and sort
    condition_values_deduped_sorted = sorted(set(condition_values))
    # Create a dictionary of condition_value: index if we're wanting to use this condition_value
    return {v: i for i, v in enumerate(condition_values_deduped_sorted)}


def get_special_tokens_for_condition(
        condition: str,
        condition_mapping: dict = None,
) -> list[str]:
    """Given a condition, generate special tokens for all observed values that can be fed to MIDITok"""
    if condition.lower() == "ensemble":
        return ["ENSEMBLE_0", "ENSEMBLE_1"]

    # Get the mapping for this condition: idx: raw_name, i.e., {0: African Jazz, 1: African Folk, ...}
    if condition_mapping is None:
        condition_mapping = get_mapping_for_condition(condition, None)
    # Create the prefix that we'll append to the start of the token
    if isinstance(condition, str):  # i.e., "pianist" -> "PIANIST_"
        token_prefix = condition.split('_')[-1].upper() + '_'
    else:  # i.e., "artist_genres" -> "GENRES_"
        token_prefix = condition[0].split('_')[-1].upper() + '_'
    # These are of the form GENRE_0, GENRE_1, GENRE_2, GENRE_3, etc.
    #  The index values conform with the mappings obtained from `get_mapping_for_condition`
    return [token_prefix.upper() + str(v) for v in condition_mapping.values()]


if __name__ == "__main__":
    print("Genre mapping: ", get_mapping_for_condition("genres"))
    print("Artist mapping: ", get_mapping_for_condition("pianist"))
    print("Mood mapping: ", get_mapping_for_condition("moods"))
