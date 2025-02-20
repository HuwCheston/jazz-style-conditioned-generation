#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Data loader and collator modules"""

import os
from collections.abc import Mapping
from typing import Any

import torch
from loguru import logger
from miditok.pytorch_data import DatasetMIDI, DataCollator
from torch.utils.data import DataLoader

from jazz_style_conditioned_generation import utils
from jazz_style_conditioned_generation.data import conditions as cond


class DatasetMIDICondition(DatasetMIDI):
    """Superclass of MIDITok dataset that grabs condition tokens for a track"""

    def __init__(
            self,
            condition_mapping: dict,
            combine_artist_and_album_tags: bool = False,
            *args,
            **kwargs,
    ) -> None:
        # This should be a dictionary of e.g., {"genre": {"African Jazz": 0, "African Folk": 1}, "mood": {}}, etc.
        self.condition_mappings = condition_mapping
        cond.validate_conditions(list(self.condition_mappings.keys()))  # raises ValueError on invalid condition
        self.conditions = list(self.condition_mappings.keys())
        # If this is True, will use tags for BOTH artist and album for mood, genre, and them
        #  Otherwise, will use album tags by default, falling back to artist tags if these are not available
        self.combine_artist_and_album_tags = combine_artist_and_album_tags
        # Initialise the MIDITok dataset
        super().__init__(*args, **kwargs)

    @staticmethod
    def load_metadata(track_filepath) -> dict:
        # Load in the metadata JSON for this track
        metadata_path = os.path.join(os.path.dirname(track_filepath), "metadata_tivo.json")
        return utils.read_json_cached(metadata_path)

    def get_conditions(self, metadata: dict) -> list[str]:
        """Return a list of special tokens associated with a given MIDI file corresponding to desired conditioning"""
        results = {}
        # i.e., "genres", "moods", "pianist" ... depending on class attributes
        for condition in self.conditions:
            if condition == "ensemble":
                continue  # not contained in metadata dict, we'll get this later from the filename itself
            condition = condition.lower()
            # If we want to use both ARTIST and ALBUM level tags
            # We can just grab these directly from the metadata dictionary
            matching_keys = [k for k in metadata.keys() if condition in k]
            # OTHERWISE, we want to use just the tags from the ALBUM
            if not self.combine_artist_and_album_tags:
                # However, not all albums have tags
                if f"album_{condition}" in metadata.keys():
                    if len(metadata[f"album_{condition}"]) > 0:
                        matching_keys = [f"album_{condition}"]
                    # Otherwise, we fall back to artist level tags
                    else:
                        matching_keys = [f"artist_{condition}"]
                        logger.warning(f'No album tags for condition {condition}, track {metadata["fname"]}, '
                                       f'falling back on artist tags with keys {matching_keys}!')
            # Raw tags for this track, e.g. [Sophisticated, Relaxed, Frantic] for mood
            condition_values = []
            try:
                condition_values = [cond.get_inner_json_values(metadata, k) for k in matching_keys]
                condition_values = [x for xs in condition_values for x in xs]
            except KeyError:
                logger.warning(f"Couldn't get keys in {matching_keys}! Tokens will be empty!")
                condition_values = []
            else:
                if len(condition_values) == 0:
                    logger.warning(f"No values for keys {matching_keys}! Tokens will be empty")
            finally:
                # Remove any values which we've specified in the EXCLUDE dictionary
                results[condition] = [cv for cv in condition_values if cv not in cond.EXCLUDE[condition]]
        # Drop duplicates and sort token list alphabetically
        return results

    def conditions_to_tokens(self, condition_results: list[dict]):
        """Convert a dictionary of {condition: [value1, value2]} to tokens"""
        results = []
        for condition in self.conditions:
            if condition == "ensemble":
                continue  # not contained in metadata dict, we'll get this later from the filename itself

            condition_res = condition_results[condition]
            # Indices of all tags for this track relative to all tags overall, e.g. [0, 1, 5]
            mapping = self.condition_mappings[condition]
            condition_values_mapped = [mapping[cv] for cv in condition_res]
            # Special tokens for all tags for this track, e.g. [MOOD_1, MOOD_2, MOOD_3]
            special_tokens = cond.get_special_tokens_for_condition(condition, self.condition_mappings[condition])
            condition_special_tokens = [special_tokens[cv] for cv in condition_values_mapped]
            results.extend(condition_special_tokens)
        return results

    @staticmethod
    def add_special_tokens_to_input(
            input_ids: torch.tensor,
            special_tokens: torch.tensor,
            insert_idx: int = 1
    ) -> torch.tensor:
        """Adds in special tokens to an input at the given `insert_idx`"""
        return torch.cat([input_ids[:insert_idx], special_tokens, input_ids[insert_idx:]])

    @staticmethod
    def get_ensemble_context_token(filename):
        dataset_name = filename.split(os.path.sep)[-3]
        if dataset_name == "jtd":
            return "ENSEMBLE_0"
        elif dataset_name == "pijama":
            return "ENSEMBLE_1"
        else:
            raise ValueError(f"Expected dataset_name to be either 'jtd' or 'pijama', but got {dataset_name}!")

    def __getitem__(self, idx: int) -> dict[str, torch.LongTensor]:
        """Return the `idx` elements of the dataset with corresponding conditions"""
        # TODO: in reality, input will probably not be a full file, but a chunk
        # Load the metadata JSON file for this track
        filename = self.files_paths[idx]
        metadata = self.load_metadata(filename)
        # This is a dictionary of {condition: [value1, value2]}, i.e. {"moods": ["Sophisticated", "Relaxed", "Frantic"]}
        conditions_for_track = self.get_conditions(metadata)
        # This is a list of special tokens, i.e. ["GENRES_28", "GENRES_61", "PIANIST_15"]
        condition_tokens_for_track = self.conditions_to_tokens(conditions_for_track)
        # If required, add in an additional ensemble context token relating to dataset (jtd or pijama)
        if "ensemble" in self.conditions:
            condition_tokens_for_track.append(self.get_ensemble_context_token(filename))
        # These are the special tokens converted to integer indices, that we can use in training the model
        condition_token_idxs = torch.tensor([self.tokenizer[c] for c in condition_tokens_for_track])
        assert [self.tokenizer[c.item()] for c in condition_token_idxs] == condition_tokens_for_track  # sanity check
        # Process the input MIDI to get token idxs
        to_collator = super().__getitem__(idx)
        # Add the special tokens in at the beginning of the sequence (after BOS)
        to_collator["input_ids"] = self.add_special_tokens_to_input(to_collator["input_ids"], condition_token_idxs)
        to_collator["labels"] = condition_token_idxs
        # A separate dictionary of metadata, that won't be passed to the collator
        not_to_collator = {
            "condition": conditions_for_track,
            "condition_tokens": condition_tokens_for_track,
            # we can maybe add more values into this?
        }
        return to_collator, not_to_collator


class CollatorMIDICondition(DataCollator):
    """Superclass of MIDITok datacollator that allows for non-tensor values in batch and output dictionary"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, batch: tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]) -> Mapping[str, Any]:
        # Split the batch into the elements we do and do not want to pass to DataCollator.__call__
        to_call, not_to_call = [i[0] for i in batch], [i[1] for i in batch]
        # Pass the required elements in: this gives us a dictionary
        proc_inputs = super().__call__(to_call)
        # We can now go and add the other elements back into this dictionary
        for k in not_to_call[0].keys():
            proc_inputs[k] = [nc[k] for nc in not_to_call]
        # This allows us to preserve e.g., the raw conditions (pianist/genre names, e.g.): should make our lives easier
        return proc_inputs


def ids_to_tokens(tokenizer, input_ids: list[int]) -> list[str]:
    """Converts a list of MidiTok input IDs to tokens (as strings)"""
    encoded_bytes = [tokenizer._model.id_to_token(i_id) for i_id in input_ids]
    decoded_tokens = [tokenizer._vocab_learned_bytes_to_tokens[byte_] for byte_ in encoded_bytes]
    return [item for sublist in decoded_tokens for item in sublist]


if __name__ == "__main__":
    from jazz_style_conditioned_generation.data.tokenizer import get_tokenizer

    # Get a tokenizer with default arguments
    token_factory = get_tokenizer()
    # Get filepaths for all MIDI files in the /data/raw/ directories
    midi_paths = utils.get_data_files_with_ext(ext="**/*.mid")
    # Define conditions and mappings
    conditions = ['genres', 'pianist', 'moods', 'themes', "ensemble"]
    condition_mappings = {c: cond.get_mapping_for_condition(c) for c in conditions}

    # Create a dataset
    dataset = DatasetMIDICondition(
        condition_mapping=condition_mappings,
        combine_artist_and_album_tags=False,
        files_paths=midi_paths,
        tokenizer=token_factory,
        max_seq_len=utils.MAX_SEQUENCE_LENGTH,
        bos_token_id=token_factory["BOS_None"],
        eos_token_id=token_factory["EOS_None"],
    )
    collator = CollatorMIDICondition(
        token_factory.pad_token_id,
        labels_pad_idx=token_factory.pad_token_id,
        copy_inputs_as_labels=False
    )
    dataloader = DataLoader(dataset, batch_size=16, collate_fn=collator)

    for b in dataloader:
        print("Batch keys: ", b.keys())
        break
