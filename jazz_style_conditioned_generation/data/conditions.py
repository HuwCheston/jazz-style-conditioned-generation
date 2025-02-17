#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Creates mappings for all conditions (performer, subgenre, mood, etc.)"""

from jazz_style_conditioned_generation import utils


def load_metadata_jsons(metadata_filepaths: list[str] = None) -> list[dict]:
    """Load all metadata JSONs corresponding to filepaths. Will use all tracks in ./data/root if no filepaths given"""
    if metadata_filepaths is None:
        metadata_filepaths = [j for j in utils.get_data_files_with_ext(ext='**/*.json') if j.endswith('_tivo.json')]
    for file in metadata_filepaths:
        yield utils.read_json_cached(file)


def get_condition_values(condition: str | list[str], metadata_dicts: list[dict]):
    """Given a condition or list of conditions, extract all values for these conditions from all metadata dicts"""
    if isinstance(condition, str):
        condition = [condition]
    res = []
    for con in condition:
        for metadata in metadata_dicts:
            condition_val = metadata[con]
            # Genre, mood, and themes are all lists of dictionaries
            if isinstance(condition_val, list):
                for condition_val_val in condition_val:
                    res.append(condition_val_val['name'])  # we also have weight values here
            # Performer is just a single string value
            else:
                res.append(condition_val)
    # i.e., [African Jazz, Bebop, Post-Bop] for genre, [Sophisticated, Relaxed, Frantic] for mood
    return res


def get_mapping_for_condition(
        condition: str | list[str],
        metadata_dicts: list[dict] = None,
        min_count: int = 1,
        exclude: list[str] = None
):
    """Gets key: idx mapping for all unique keys under a given condition (or list of conditions).

    Minimum number of appearances and keys to exclude can be specified as arguments. If a list of dictionaries is
    not provided, will scrape all "metadata_tivo.json" files found in ./data/raw/<dataset>/<track> recursively.
    """
    if metadata_dicts is None:
        metadata_dicts = load_metadata_jsons(None)
    if exclude is None:
        exclude = []
    # Get the "raw" values for this condition (i.e., a list of all genres for all tracks, with duplicates)
    condition_values = get_condition_values(condition, metadata_dicts)
    # Remove duplicates and sort
    condition_values_deduped_sorted = sorted(set(condition_values))
    # Create a dictionary of condition_value: index if we're wanting to use this condition_value
    return {
        v: i for i, v in enumerate(condition_values_deduped_sorted)
        if len([r for r in condition_values if r == v]) >= min_count and v not in exclude
    }


def get_counts_for_conditions(
        condition: str | list[str],
        metadata_dicts: list[dict] = None,
) -> dict:
    """Given a condition, return the number of times this can be seen in the metadata files"""
    if metadata_dicts is None:
        metadata_dicts = load_metadata_jsons(None)
    # Get the "raw" values for this condition (i.e., a list of all genres for all tracks, with duplicates)
    condition_values = get_condition_values(condition, metadata_dicts)
    # Remove duplicates and sort
    condition_values_deduped_sorted = sorted(set(condition_values))
    # Create a dictionary of condition_value: number_of_occurrences
    return {v: len([r for r in condition_values if r == v]) for v in condition_values_deduped_sorted}


def get_special_tokens_for_condition(condition: str, token_prefix: str = "GENRE_") -> list[str]:
    """Given a condition, generate special tokens for all observed values that can be fed to MIDITok"""
    if not token_prefix.endswith('_'):
        token_prefix += '_'
    condition_mapping = get_mapping_for_condition(condition=condition)
    # These are of the form GENRE_0, GENRE_1, GENRE_2, GENRE_3, etc.
    #  The index values conform with the mappings obtained from `get_mapping_for_condition`
    return [token_prefix.upper() + str(v) for v in condition_mapping.values()]


# Define these as variables here so we can access them easily without having to re-run any costly loading functions
PERFORMER_MAPPING = get_mapping_for_condition("pianist")
GENRE_MAPPING = get_mapping_for_condition(["artist_genres", "album_genres"])
MOOD_MAPPING = get_mapping_for_condition(["artist_moods", "album_moods"])
THEME_MAPPING = get_mapping_for_condition("album_themes")
