#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Creates mappings for all conditions (performer, subgenre, mood, etc.)"""

import os

from jazz_style_conditioned_generation import utils

# These are the only conditions we'll accept values for
ACCEPT_CONDITIONS = ["moods", "genres", "pianist", "themes"]
# Each list should be populated with values for a condition that we don't want to use
EXCLUDE = {
    "genres": [
        # Nearly every track could be described as one of these genres
        "Jazz Instrument",
        "Jazz",
        "Piano Jazz",
        "Keyboard",
        "Solo Instrumental",
        "Improvisation",
        # These tags seem incorrect given the type of music we know to be in the dataset
        "Big Band",
        "Choral",
        "Electronic",
        "Guitar Jazz",
        "Modern Big Band",
        "Orchestral",
        "Saxophone Jazz",
        "Symphony",
        "Trumpet Jazz",
        "Vibraphone/Marimba Jazz",
        "Vocal",
        "Vocal Music",
        "Vocal Pop",
    ],
    "moods": [],
    "pianist": [],
    "themes": [],
}
MERGE = {
    "genres": {
        "Adult Alternative Pop/Rock": "Pop/Rock",
        "Alternative/Indie Rock": "Pop/Rock",
        "African Folk": "African",
        "African Jazz": "African",
        "African Traditions": "African",
        "American Popular Song": "Pop/Rock",
        "Avant-Garde Jazz": "Avant-Garde",
        "Ballet": "Stage & Screen",
        "Brazilian Pop": "Brazilian",
        "Brazilian Jazz": "Brazilian",
        "Brazilian Traditions": "Brazilian",
        "Black Gospel": "Gospel",
        "Cast Recordings": "Stage & Screen",
        "Calypso": "Caribbean",
        "Caribbean Traditions": "Caribbean",
        "Central/West Asian Traditions": "Asian",
        "Chamber Jazz": "Chamber",
        "Chamber Music": "Chamber",
        "Christmas": "Holiday",
        "Classical Crossover": "Classical",
        "Concerto": "Classical",
        "Contemporary Jazz": "Modern Jazz",
        "Club/Dance": "Electronic",
        "Cuban Jazz": "Afro-Cuban Jazz",
        "Electro": "Electronic",
        "European Folk": "European",
        "French": "European",
        "Film Score": "Stage & Screen",
        "Film Music": "Stage & Screen",
        "Global Jazz": "International",
        "Holidays": "Holiday",
        "Jazz Blues": "Blues",
        "Jazz-Funk": "Funk",
        "Jazz-Pop": "Pop/Rock",
        "Latin": "Latin Jazz",
        "Modern Composition": "Modern Jazz",
        "Modern Creative": "Modern Jazz",
        "Modern Free": "Modern Jazz",
        "Musical Theater": "Stage & Screen",
        "New Orleans Jazz Revival": "New Orleans Jazz",
        "Original Score": "Stage & Screen",
        "Piano/Easy Listening": "Easy Listening",
        "Show Tunes": "Stage & Screen",
        "Show/Musical": "Stage & Screen",
        "Soundtracks": "Stage & Screen",
        "South African Folk": "African",
        "South American Traditions": "Southern American",
        "Spirituals": "Gospel",
        "Spy Music": "Stage & Screen",
        "Traditional Pop": "Pop/Rock",
        "Western European Traditions": "European",
        "Venezuelan": "Southern American"
    },
    "moods": {},
    "pianist": {},
    "themes": {},
}


def validate_conditions(conditions: list[str] | str) -> None:
    """Validates a single condition or list of conditions, raises ValuError for invalid conditions"""
    # Allows for a single string as input
    if isinstance(conditions, str):
        conditions = [conditions]
    for condition in conditions:
        if condition.lower() not in ACCEPT_CONDITIONS:
            raise ValueError(f'expected `condition` to be in {", ".join(ACCEPT_CONDITIONS)} but got {condition}')


def load_tivo_metadata() -> list[dict]:
    """Load all metadata JSONs from data/root and references/tivo_artist_metadata"""
    metadata_filepaths = []
    # Get the JSON files from data/raw (i.e., for individual tracks)
    for dataset in utils.DATASETS_WITH_TIVO:
        dataset_jsons = [
            j for j in utils.get_data_files_with_ext(
                dir_from_root=os.path.join("data/raw", dataset), ext="**/*.json"
            )
        ]
        metadata_filepaths.extend([j for j in dataset_jsons if j.endswith("_tivo.json")])
    # Get the JSON files from references/tivo_artist_metadata (i.e., for each artist)
    for artist in os.listdir(os.path.join(utils.get_project_root(), "references/tivo_artist_metadata")):
        metadata_filepaths.append(os.path.join(utils.get_project_root(), "references/tivo_artist_metadata", artist))
    # Iterate through and read all the JSONs (with a cache)
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


def validate_condition_values(condition_values: list[str] | str, condition_key: str) -> list[str]:
    """Validates values for a given condition by merging similar entries, removing invalid ones, etc."""
    new_values = []
    # Allow both single strings and list of strings as input
    if isinstance(condition_values, str):
        condition_values = [condition_values]
    for c in condition_values:
        # Merge similar values together
        if c in MERGE[condition_key].keys():
            c = MERGE[condition_key][c]
        # Remove any values for this condition that we don't want to use
        if c not in EXCLUDE[condition_key]:
            new_values.append(c)
    # Remove duplicates and sort
    return list(sorted(set(new_values)))


def get_condition_special_tokens(condition: str, metadata_dicts: list[dict] | dict = None):
    """Given a condition in ACCEPT_CONDITIONS, extract all observed values from all metadata dicts"""
    # Check that the condition we've passed in is valid
    validate_conditions(condition)
    # Allow a single dictionary as input
    if isinstance(metadata_dicts, dict):
        metadata_dicts = [metadata_dicts]
    # Otherwise, if we haven't passed any metadata in, just grab this from the entire dataset
    elif metadata_dicts is None:
        metadata_dicts = load_tivo_metadata()
    # We'll skip over these values
    condition_values = []
    for metadata in metadata_dicts:
        # This will give us e.g., ["artist_genres", "album_genres"] for the input "genres"
        accept_keys = [k for k in metadata.keys() if condition.lower() in k.lower()]
        # Iterate over all of these keys and get the required value from the metadata dictionary
        for accept_key in accept_keys:
            condition_values.extend(get_inner_json_values(metadata, accept_key))
    # Merge similar values, remove exclude values + duplicates, and sort alphabetically
    condition_values_validated_deduped_sorted = validate_condition_values(condition_values, condition)
    # Return a mapping of condition value: special token
    return {
        s: f'{condition.upper()}_{utils.remove_punctuation(s).replace(" ", "")}'
        for s in condition_values_validated_deduped_sorted
    }


if __name__ == "__main__":
    for condition_type in ACCEPT_CONDITIONS:
        sts = get_condition_special_tokens(condition_type)
        print(f"{condition_type.title()} number of unique values: {len(list(sts.keys()))}")
        print(f"{condition_type.title()} unique values: {list(sts.keys())}")
        print(f"{condition_type.title()} special tokens: {list(sts.values())}")
        print('\n')
