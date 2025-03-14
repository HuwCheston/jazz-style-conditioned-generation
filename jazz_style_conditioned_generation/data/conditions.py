#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Creates mappings for all conditions (performer, subgenre, mood, etc.)"""

import os

import numpy as np
from miditok import MusicTokenizer

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
    "pianist": [
        "JJA Pianist 1",
        "JJA Pianist 2",
        "JJA Pianist 3",
        "JJA Pianist 4",
        "JJA Pianist 5",
        "Doug McKenzie"
        # TODO: consider adding pianists who have fewer than N tracks here
    ],
    "themes": [],
}
# Xu et al. (2023): 1.5 million musescore MIDI files, yet only 20 genre tags. We want to reduce to sub-30 genres.
MERGE = {
    "genres": {
        "Adult Alternative Pop/Rock": "Pop/Rock",
        "Alternative/Indie Rock": "Pop/Rock",
        "African Folk": "African",
        "African Jazz": "African",
        "African Traditions": "African",
        "Afro-Cuban Jazz": "African",
        "American Popular Song": "Pop/Rock",
        "Avant-Garde": "Avant-Garde Jazz",
        # "Avant-Garde Jazz": "Avant-Garde",
        "Ballet": "Stage & Screen",
        "Brazilian": "Southern American",
        "Brazilian Pop": "Southern American",
        "Brazilian Jazz": "Southern American",
        "Brazilian Traditions": "Southern American",
        "Black Gospel": "Gospel & Religious",
        "Boogie-Woogie": "Blues",
        "Cast Recordings": "Stage & Screen",
        "Calypso": "Caribbean",
        "Caribbean Traditions": "Caribbean",
        "Central/West Asian Traditions": "Asian",
        "Chamber": "Classical & Chamber",
        "Chamber Jazz": "Classical & Chamber",
        "Chamber Music": "Classical & Chamber",
        "Christmas": "Gospel & Religious",
        "Classical": "Classical & Chamber",
        "Classical Crossover": "Classical & Chamber",
        "Cool": "Cool Jazz",
        "Concerto": "Classical & Chamber",
        "Contemporary Jazz": "Modern Jazz",
        "Crossover Jazz": "Easy Listening",
        "Club/Dance": "Electronic",
        "Cuban Jazz": "Caribbean",
        "Dixieland": "Early & Trad Jazz",
        "Early Jazz": "Early & Trad Jazz",
        "Electro": "Electronic",
        "European Folk": "European",
        "French": "European",
        "Free Improvisation": "Free Jazz",
        "Film Score": "Stage & Screen",
        "Film Music": "Stage & Screen",
        # "Funk": "Pop/Rock",
        "Global Jazz": "International",
        "Gospel": "Gospel & Religious",
        "Holidays": "Gospel & Religious",
        "Holiday": "Gospel & Religious",
        "Highlife": "African",
        "Jazz Blues": "Blues",
        "Jazz-Funk": "Funk",
        "Jazz-Pop": "Pop/Rock",
        "Latin": "Latin Jazz",
        "Lounge": "Easy Listening",
        "Mainstream Jazz": "Straight-Ahead Jazz",
        "Modal Music": "Modal Jazz",
        "Modern Composition": "Modern Jazz",
        "Modern Creative": "Modern Jazz",
        "Modern Free": "Free Jazz",
        "Musical Theater": "Stage & Screen",
        "Neo-Bop": "Post-Bop",
        "New Age": "Easy Listening",
        "New Orleans Jazz Revival": "Early & Trad Jazz",
        "New Orleans Jazz": "Early & Trad Jazz",
        "Original Score": "Stage & Screen",
        "Piano/Easy Listening": "Easy Listening",
        "Progressive Jazz": "Modern Jazz",
        "Ragtime": "Early & Trad Jazz",
        "Religious": "Gospel & Religious",
        "Show Tunes": "Stage & Screen",
        "Show/Musical": "Stage & Screen",
        "Smooth Jazz": "Easy Listening",
        "Soundtracks": "Stage & Screen",
        "South African Folk": "African",
        "Southern African": "African",
        "South American Traditions": "Southern American",
        "Spirituals": "Gospel & Religious",
        "Spy Music": "Stage & Screen",
        "Standards": "Straight-Ahead Jazz",
        "Stride": "Early & Trad Jazz",
        "Swing": "Early & Trad Jazz",
        "Third Stream": "Classical & Chamber",
        "Trad Jazz": "Early & Trad Jazz",
        "Traditional Pop": "Pop/Rock",
        "Township Jazz": "African",
        "West Coast Jazz": "Cool Jazz",
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


def validate_condition_values(
        condition_values: list[tuple[str, int]],
        condition_name: str
) -> list[tuple[str, int]]:
    """Validates values for a given condition by merging similar entries, removing invalid ones, etc."""
    validated = {}
    for value, weight in condition_values:
        # Merge a value with its "master" key (i.e., Show Tunes -> Stage & Screen, Soundtrack -> Stage & Screen)
        if value in MERGE[condition_name]:
            value = MERGE[condition_name][value]
        # Skip over value that we don't want to use
        if value not in EXCLUDE[condition_name]:
            # This ensures that we only store the HIGHEST weight for any value
            if value not in validated.keys() or weight > validated[value]:
                validated[value] = weight
    # Sanity check that none of our values should now be duplicates
    assert len(set(validated.keys())) == len(validated.keys())
    # Sort the values by their weight, in descending order (highest weight first)
    return sorted(list(validated.items()), key=lambda x: x[1], reverse=True)


def _get_pianist_genres(pianist_name: str) -> list[tuple[str, int]]:
    """Get the genres & weights associated with the PIANIST playing on a track (not the track itself)"""
    pianist_metadata = os.path.join(
        utils.get_project_root(),
        "references/tivo_artist_metadata",
        pianist_name.replace(" ", "") + ".json"
    )
    # If we have metadata for the pianist, grab the associated genres
    if os.path.isfile(pianist_metadata):
        # Read the metadata for the pianist
        pianist_metadata_dict = utils.read_json_cached(pianist_metadata)
        # If we have genres for the pianist
        if len(pianist_metadata_dict["genres"]) > 0:
            genres = [(x["name"], x["weight"]) for x in pianist_metadata_dict["genres"]]
            return validate_condition_values(genres, "genres")
        # Otherwise, return an empty list
        else:
            return []
    # This will trigger if we SHOULD have metadata for the current pianist, but we can't find the file
    elif pianist_name not in EXCLUDE["pianist"]:
        raise FileNotFoundError(f"Could not find metadata file at {pianist_metadata}!")
    # This will trigger if we shouldn't have metadata for the pianist: silently return an empty list
    else:
        return []


def _get_track_genres(track_metadata_dict: dict) -> list[tuple[str, int]]:
    """Get the genres & weights associated with a track"""
    # Grab the genres associated with the track
    if len(track_metadata_dict) > 0:
        genres = [(x["name"], x["weight"]) for x in track_metadata_dict["genres"]]
        # Validate the genres: remove any duplicates/values we don't want
        return validate_condition_values(genres, "genres")
    else:
        return []


def get_genre_tokens(track_metadata_dict: dict, tokenizer: MusicTokenizer, n_genres: int = None):
    """Gets tokens for a track's genres: either from the track itself, or (if none found) from the artist"""
    # Check that we've added pianist tokens to our tokenizer
    assert len([i for i in tokenizer.vocab.keys() if "GENRES" in i]) > 0, "Genre tokens not added to tokenizer!"
    # Try and get the tokens for the TRACK first
    genres = _get_track_genres(track_metadata_dict)
    # If we don't have any genres for the TRACK
    if len(genres) == 0:
        # Try and get them for the PIANIST
        pianist_genres = _get_pianist_genres(track_metadata_dict["pianist"])
        # If we still don't have any genres, just return an empty list
        if len(pianist_genres) == 0:
            return []
        # Otherwise, we can use the genres associated with the pianist
        else:
            genres = pianist_genres
    # Remove the weight term from each tuple to get a single list
    finalised_genres = [g[0] for g in genres]
    # Subset to only get the top-N genres, if required
    if n_genres is not None:
        finalised_genres = finalised_genres[:n_genres]
    # Add the prefix to the token
    prefixed = [f'GENRES_{utils.remove_punctuation(g).replace(" ", "")}' for g in finalised_genres]
    # Sanity check that the tokens are part of the vocabulary for the tokenizer
    for pfix in prefixed:
        assert pfix in tokenizer.vocab.keys(), f"Could not find token {pfix} in tokenizer vocabulary!"
    return prefixed


def _get_similar_pianists(pianist_name: str) -> list[tuple[str, int]]:
    """Get names + weights for pianists SIMILAR to the current pianist on a track"""
    pianist_metadata = os.path.join(
        utils.get_project_root(),
        "references/tivo_artist_metadata",
        pianist_name.replace(" ", "") + ".json"
    )
    # If we have metadata for the pianist, grab the other pianists that TiVo says they are similar to
    if os.path.isfile(pianist_metadata):
        pianist_metadata_dict = utils.read_json_cached(pianist_metadata)
        all_pianists = [(x["name"], x["weight"]) for x in pianist_metadata_dict["similars"]]
        return validate_condition_values(all_pianists, "pianist")
    # This will trigger if we SHOULD have metadata for the current pianist, but we can't find the file
    elif pianist_name not in EXCLUDE["pianist"]:
        raise FileNotFoundError(f"Could not find metadata file at {pianist_metadata}!")
    # This will trigger if we DON'T have metadata for the current pianist, and we SHOULDN't have metadata
    else:
        return []


def get_pianist_tokens(track_metadata_dict: dict, tokenizer: MusicTokenizer, n_pianists: int = None) -> list[str]:
    # Check that we've added pianist tokens to our tokenizer
    assert len([i for i in tokenizer.vocab.keys() if "PIANIST" in i]) > 0, "Pianist tokens not added to tokenizer!"
    # Get the pianist FROM THIS TRACK
    track_pianist = track_metadata_dict["pianist"]
    # Get pianists that are similar to this pianist
    similar_pianists = _get_similar_pianists(track_pianist)
    # Remove the weight term
    similar_pianists = [s[0] for s in similar_pianists]
    # If we can use the track pianist
    if track_pianist not in EXCLUDE["pianist"]:
        # Subset to get only the top-N - 1 pianists if required
        if n_pianists is not None:
            similar_pianists = similar_pianists[:n_pianists - 1]
        finalised_pianists = [track_pianist] + similar_pianists
    # Otherwise, we want to keep top-N pianists
    else:
        if n_pianists is not None:
            similar_pianists = similar_pianists[:n_pianists]
        finalised_pianists = similar_pianists
    # Add the prefix to the token
    prefixed = [f'PIANIST_{utils.remove_punctuation(g).replace(" ", "")}' for g in finalised_pianists]
    # Sanity check that the tokens are part of the vocabulary for the tokenizer
    for pfix in prefixed:
        assert pfix in tokenizer.vocab.keys(), f"Could not find token {pfix} in tokenizer vocabulary!"
    return prefixed


def get_tempo_token(tempo: float, tokenizer: MusicTokenizer, _raise_on_difference_exceeding: int = 50) -> str:
    """Given a tempo for a track, get the closest tempo token from the tokenizer"""
    # Get the tempo tokens from the tokenizer
    tempo_tokens = [i for i in tokenizer.vocab.keys() if "TEMPOCUSTOM" in i]
    assert len(tempo_tokens) > 0, "Custom tempo tokens not added to tokenizer!"
    # Get tempo values as integers, rather than strings
    tempo_stripped = np.array([int(i.replace("TEMPOCUSTOM_", "")) for i in tempo_tokens])
    # Get the difference between the passed tempo and the tempo tokens used by the tokenizer
    sub = np.abs(tempo - tempo_stripped)
    # Raise an error if the closest tempo token is too far away from the actual token
    if np.min(sub) > _raise_on_difference_exceeding:
        raise ValueError(f"Closest tempo token is too far from passed tempo! Smallest difference is {sub}")
    # Get the actual tempo token
    tempo_token = tempo_tokens[np.argmin(sub)]
    # Sanity check
    assert "TEMPOCUSTOM" in tempo_token
    # Return the idx of the token, ready to be added to the sequence
    return tempo_token


def get_time_signature_token(time_signature: int, tokenizer: MusicTokenizer) -> str:
    # Get the time signature tokens from the tokenizer
    timesig_tokens = [i for i in tokenizer.vocab.keys() if "TIMESIGNATURECUSTOM" in i]
    assert len(timesig_tokens) > 0, "Custom time signature tokens not added to tokenizer!"
    # Get the corresponding token
    timesig_token = f'TIMESIGNATURECUSTOM_{time_signature}4'
    # Return the idx, ready to be added to the sequence
    if timesig_token in timesig_tokens:
        return timesig_token
    # Raise an error if the token isn't in the vocabulary (should never happen)
    else:
        raise AttributeError(f"Tokenizer does not have token {timesig_token} in vocabulary!")


if __name__ == "__main__":
    from collections import Counter

    from miditok import MIDILike
    from jazz_style_conditioned_generation.data.tokenizer import add_genres_to_vocab, add_pianists_to_vocab

    tokfactory = MIDILike()
    js_fps = utils.get_data_files_with_ext("data/raw", "**/*_tivo.json")
    add_genres_to_vocab(tokfactory, js_fps)
    add_pianists_to_vocab(tokfactory, js_fps)

    track_genres, track_pianists = [], []
    for js in js_fps:
        js_loaded = utils.read_json_cached(js)
        track_genres.extend(get_genre_tokens(js_loaded, tokfactory))
        track_pianists.extend(get_pianist_tokens(js_loaded, tokfactory))

    print("Loaded", len(set(track_genres)), "genres")
    assert len(set(track_genres)) == len([i for i in tokfactory.vocab.keys() if "GENRES" in i])
    print("Genre counts: ", Counter(track_genres))
    print("Loaded", len(set(track_pianists)), "pianists")
    assert len(set(track_pianists)) == len([i for i in tokfactory.vocab.keys() if "PIANIST" in i])
    print("Pianist counts: ", Counter(track_pianists))
