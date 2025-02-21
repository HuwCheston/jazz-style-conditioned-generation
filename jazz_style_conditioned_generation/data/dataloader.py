#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Data loader and collator modules"""

import os
import random

import torch
from miditok.pytorch_data import DatasetMIDI, DataCollator
from torch.utils.data import DataLoader

from jazz_style_conditioned_generation import utils
from jazz_style_conditioned_generation.data import conditions as cond

DATA_DIR = os.path.join(utils.get_project_root(), "data")

AUGMENT_PROB = 0.5
PITCH_AUGMENT_RANGE = (-(utils.OCTAVE // 2), (utils.OCTAVE // 2))
VELOCITY_AUGMENT_RANGE = (0.8, 1.2)
DURATION_AUGMENT_RANGE = (12, 12)


class DatasetMIDICondition(DatasetMIDI):
    """Superclass of MIDITok dataset that grabs condition tokens for a track"""

    def __init__(
            self,
            condition_mapping: dict,
            combine_artist_and_album_tags: bool = False,
            n_clips: int = None,
            skip_conditioning: bool = False,
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
        self.skip_conditioning = skip_conditioning
        # Initialise the MIDITok dataset
        super().__init__(*args, **kwargs)
        if n_clips is not None:
            random.shuffle(self.files_paths)
            self.files_paths = self.files_paths[:n_clips]

    @staticmethod
    def load_metadata(chunk_filepath) -> dict:
        # Load in the metadata JSON for this track
        track_filepath = chunk_filepath.replace("/chunks/", "/raw/")
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
                        # logger.warning(f'No album tags for condition {condition}, track {metadata["fname"]}, '
                        #                f'falling back on artist tags with keys {matching_keys}!')
            # Raw tags for this track, e.g. [Sophisticated, Relaxed, Frantic] for mood
            condition_values = []
            try:
                condition_values = [cond.get_inner_json_values(metadata, k) for k in matching_keys]
                condition_values = [x for xs in condition_values for x in xs]
            except KeyError:
                condition_values = []
            finally:
                # if len(condition_values) == 0:
                #     logger.warning(f"No values for keys {matching_keys}! Tokens will be empty")
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
    def get_ensemble_context_token(filename):
        dataset_name = filename.split(os.path.sep)[-3]
        if dataset_name == "jtd":
            return "ENSEMBLE_0"
        elif dataset_name == "pijama":
            return "ENSEMBLE_1"
        else:
            raise ValueError(f"Expected dataset_name to be either 'jtd' or 'pijama', but got {dataset_name}!")

    # def augment_score(self, score: Score) -> Score:
    #     if random.uniform(0, 1) < AUGMENT_PROB:
    #         pitch_augment = random.randrange(*PITCH_AUGMENT_RANGE)
    #         velocity_augment = random.randrange(*VELOCITY_AUGMENT_RANGE)
    #         duration_augment = random.uniform(*DURATION_AUGMENT_RANGE)
    #
    # def getitem_with_augmentation(self, idx: int):
    #     """A copy of super().__getitem__ that allows for data augmentation of a `symusic.Score` object"""
    #     labels = None
    #
    #     # Already pre-tokenized
    #     if self.pre_tokenize:
    #         token_ids = self.samples[idx]
    #         if self.func_to_get_labels is not None:
    #             labels = self.labels[idx]
    #
    #     # Tokenize on the fly
    #     else:
    #         # The tokenization steps are outside the try bloc as if there are errors,
    #         # we might want to catch them to fix them instead of skipping the iteration.
    #         try:
    #             score = Score(self.files_paths[idx])
    #         except SCORE_LOADING_EXCEPTION:
    #             item = {self.sample_key_name: None}
    #             if self.func_to_get_labels is not None:
    #                 item[self.labels_key_name] = labels
    #             return item
    #
    #         # DATA AUGMENTATION COMES HERE
    #
    #         tseq = self._tokenize_score(score)
    #         # If not one_token_stream, we only take the first track/sequence
    #         token_ids = tseq.ids if self.tokenizer.one_token_stream else tseq[0].ids
    #         if self.func_to_get_labels is not None:
    #             # tokseq can be given as a list of TokSequence to get the labels
    #             labels = self.func_to_get_labels(score, tseq, self.files_paths[idx])
    #             if not isinstance(labels, torch.LongTensor):
    #                 labels = torch.LongTensor([labels] if isinstance(labels, int) else labels)
    #
    #     item = {self.sample_key_name: torch.LongTensor(token_ids)}
    #     if self.func_to_get_labels is not None:
    #         item[self.labels_key_name] = labels
    #
    #     return item

    def __getitem__(self, idx: int) -> dict[str, torch.LongTensor]:
        """Return the `idx` elements of the dataset with corresponding conditions"""
        # Process the input MIDI to get token idxs
        processed = super().__getitem__(idx)
        if self.skip_conditioning:
            return processed

        # Grab the first token from the processed input
        first_token = processed["input_ids"][0].item()
        # If this is the beginning of the sequence, we don't need to add our conditioning tokens
        if first_token != self.tokenizer["BOS_None"]:
            # We can just return the sequence
            return processed
        initial_size = processed["input_ids"].size(0)
        # Otherwise, we need to get the conditioning tokens
        filename = self.files_paths[idx]
        # Should always be the first chunk from any track
        assert utils.get_chunk_number_from_filepath(filename) == 0
        # Load the metadata JSON file for this track
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
        # Add the special tokens in at the beginning of the sequence (after BOS)
        processed["input_ids"] = utils.add_to_tensor_at_idx(processed["input_ids"], condition_token_idxs)
        assert processed["input_ids"].size(0) > initial_size
        return processed


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
        n_clips=100,
        condition_mapping=condition_mappings,
        combine_artist_and_album_tags=False,
        files_paths=midi_paths,
        tokenizer=token_factory,
        max_seq_len=utils.MAX_SEQUENCE_LENGTH,
        bos_token_id=token_factory["BOS_None"],
        eos_token_id=token_factory["EOS_None"],
    )
    collator = DataCollator(
        token_factory.pad_token_id,
        copy_inputs_as_labels=True,
        shift_labels=True
    )
    dataloader = DataLoader(dataset, batch_size=16, collate_fn=collator)

    for b in dataloader:
        print("Batch keys: ", b.keys())
        break
