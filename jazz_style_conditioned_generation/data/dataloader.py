#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Data loader and collator modules"""

import os
import random
from copy import deepcopy

import numpy as np
import torch
from miditok.constants import SCORE_LOADING_EXCEPTION
from miditok.data_augmentation import augment_score
from symusic import Score
from tqdm import tqdm

from jazz_style_conditioned_generation import utils

DATA_DIR = os.path.join(utils.get_project_root(), "data")

DEFAULT_AUGMENTATION_PROB = 0.5
PITCH_AUGMENT_RANGE = range(-3, 3)  # as in Music Transformer
DURATION_AUGMENT_RANGE = [0.95, 0.975, 1.0, 1.025, 1.05]  # as in Music Transformer


def get_pitch_augmentation_value(score: Score, pitch_augmentation_range: list) -> int:
    """Gets a pitch augmentation value that can be applied to a Score without exceeding the limits of the keyboard"""
    # Get the minimum and maximum pitch from the score
    min_pitch, max_pitch = utils.get_pitch_range(score)
    # Default values
    pitch_augment, min_pitch_augmented, max_pitch_augmented = 0, 0, 1000
    # Keep iterating until we have an acceptable pitch augmentation value
    while min_pitch_augmented < utils.MIDI_OFFSET or max_pitch_augmented > utils.MIDI_OFFSET + utils.PIANO_KEYS:
        # Get a possible pitch augment value
        pitch_augment = np.random.choice(pitch_augmentation_range, 1)
        # Add this to the min and max pitch
        min_pitch_augmented = min_pitch + pitch_augment
        max_pitch_augmented = max_pitch + pitch_augment
    return pitch_augment


def data_augmentation(
        score: Score,
        pitch_augmentation_range: list = None,
        duration_augmentation_range: list = None
) -> Score:
    """Applies pitch and duration augmentation to a Score object"""
    if pitch_augmentation_range is None:
        pitch_augmentation_range = PITCH_AUGMENT_RANGE
    if duration_augmentation_range is None:
        duration_augmentation_range = DURATION_AUGMENT_RANGE

    # Get augmentation values
    pitch_augment = get_pitch_augmentation_value(score, pitch_augmentation_range)
    # We can augment pitch directly with the MIDItok function
    augmented = augment_score(score, pitch_offset=pitch_augment, augment_copy=True, duration_offset=0, )
    # We need to use symusic to adjust the durations
    duration_augment = np.random.choice(duration_augmentation_range)
    return augmented.adjust_time(
        [augmented.start(), augmented.end()],
        [augmented.start(), int(augmented.end() * duration_augment)],
        inplace=False
    )


def validate_midi_paths(midi_filepaths: list[str]):
    """Validates that all midi paths exist on disk"""
    for file in midi_filepaths:
        assert os.path.isfile(file), f"File {file} does not exist!"
        assert file.endswith(".mid"), f"File {file} does not appear to be a MIDI file!"


def add_beginning_and_ending_tokens_to_sequence(
        tokseq_ids: list[int],
        bos_token_id: int,
        eos_token_id: int
) -> list[int]:
    """Adds beginning and ending of sequence tokens to a COMPLETE track"""
    # Add BOS and EOS tokens in
    tokseq_ids.insert(0, bos_token_id)
    tokseq_ids.insert(len(tokseq_ids), eos_token_id)
    # Sanity check everything is in its right place
    assert tokseq_ids[0] == bos_token_id
    assert tokseq_ids[-1] == eos_token_id
    return tokseq_ids


def pad_sequence(
        sequence: list[int],
        desired_len: int,
        pad_token_id: int,
        right_pad: bool = True
) -> list[int]:
    """(Right- or left-) pads a sequence to desired length"""
    # Create an array of padding tokens
    x = [pad_token_id for _ in range(desired_len)]
    # Replace the initial tokens with our sequence
    if right_pad:
        x[:len(sequence)] = sequence
    else:
        x[-len(sequence):] = sequence
    return x


def randomly_slice_sequence(
        sequence: list[int],
        desired_len: int,
) -> list[int]:
    """Randomly slices a sequence into desired length"""
    assert len(sequence) > desired_len
    end_range = len(sequence) - desired_len
    start = random.randint(0, end_range)
    end = start + desired_len
    return sequence[start:end]


class DatasetMIDIRandomChunk:
    """Dataloader that returns a random chunk of `max_seq_len` for a single track"""
    def __init__(
            self,
            tokenizer,
            files_paths: list[str],
            max_seq_len: int,
            do_augmentation: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.do_augmentation = do_augmentation
        validate_midi_paths(files_paths)
        self.files_paths = files_paths

    def __len__(self):
        return len(self.files_paths)

    def chunk_sequence(self, sequence: list[int]) -> tuple[list[int], list[int]]:
        # We need an overlap of 1 to shift the targets for autoregressive modelling
        full_seq_length = self.max_seq_len + 1
        # If the full sequence is too short
        if len(sequence) < full_seq_length:
            # We can just pad the sequence
            x = pad_sequence(sequence, desired_len=full_seq_length, pad_token_id=self.tokenizer.pad_token_id)
        # Randomly chunking the sequence
        else:
            x = randomly_slice_sequence(sequence, desired_len=full_seq_length)
        # Shift labels by one for autoregression
        targets = x[1:]
        x = x[:-1]  # remove the final token to get the desired length
        assert len(targets) == len(x) == self.max_seq_len
        return x, targets

    def __getitem__(self, idx: int) -> dict[str, torch.LongTensor]:
        # Get the filepath for the corresponding track
        fp = self.files_paths[idx]
        # Convert into a Symusic Score object
        try:
            score = Score(fp)
        except SCORE_LOADING_EXCEPTION:
            return {"input_ids": None, "labels": None}
        # Perform data augmentation on the score object if required
        if self.do_augmentation:
            score = data_augmentation(score)
        # Preprocess the music file
        score = self.tokenizer.preprocess_score(score)
        # Tokenize it
        tokseq = self.tokenizer.encode(score, no_preprocess_score=True)
        # Add BOS and EOS tokens in
        # MIDITok only adds a BOS and EOS token at the beginning and ending of the track
        temp_ids = deepcopy(tokseq[0].ids)
        # TODO: confirm that we don't need BOS and EOS tokens at the end of EVERY sequence
        tokseq_ids = add_beginning_and_ending_tokens_to_sequence(
            temp_ids, self.tokenizer["BOS_None"], self.tokenizer["EOS_None"]
        )
        # TODO: here is where we can add conditioning tokens
        # TODO: potentially remove bar/tempo tokens here (they have no meaning)
        input_ids, targets = self.chunk_sequence(tokseq_ids)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(targets, dtype=torch.long)
        }

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"Random chunk dataloader with {len(self.files_paths)} tracks/chunks."


class DatasetMIDIExhaustive:
    """Dataset that returns all `max_seq_len` chunks from a single track"""

    def __init__(
            self,
            tokenizer,
            files_paths: list[str],
            max_seq_len: int,
            do_augmentation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        if do_augmentation:
            raise NotImplementedError("Data augmentation not implemented for exhaustive MIDI loader")
        validate_midi_paths(files_paths)
        # [filename1, filename2, ...]
        self.files_paths = files_paths
        # [(filename1, chunk1), (filename1, chunk2), (filename2, chunk1), ...]
        self.chunk_paths_and_idxs = list(self.get_chunks_per_track())

    def chunker(self, score: Score) -> list[list[int]]:
        """Chunks a symusic.Score object into chunks of `max_seq_len` size"""
        # Preprocess the music file
        score = self.tokenizer.preprocess_score(score)
        # Tokenize it
        tokseq = self.tokenizer.encode(score, no_preprocess_score=True)
        temp_ids = deepcopy(tokseq[0].ids)
        # Add beginning and ending of sequence tokens in
        tokseq = add_beginning_and_ending_tokens_to_sequence(
            temp_ids, self.tokenizer["BOS_None"], self.tokenizer["EOS_None"]
        )
        # Split into chunks of `max_seq_len` size, plus 1 for autoregressive label shifting
        return [t.tolist() for t in torch.split(torch.tensor(tokseq), self.max_seq_len + 1)]

    def get_chunks_per_track(self) -> tuple[str, int]:
        """For every track in the list of file paths, gets the number of chunks"""
        # Iterate over every file
        for file in tqdm(self.files_paths, desc='Getting track chunks...'):
            # Open file as a symusic score object
            score = Score(file)
            # Convert into chunks
            chunked_file = self.chunker(score)
            # Yield tuples of (filename, chunk_idx)
            for chunk_idx in range(len(chunked_file)):
                yield file, chunk_idx

    def __getitem__(self, idx: int) -> dict[str, torch.LongTensor]:
        # Get the filepath and idx of the desired chunk
        fp, chunk_idx = self.chunk_paths_and_idxs[idx]
        # Convert into a Symusic Score object
        try:
            score = Score(fp)
        except SCORE_LOADING_EXCEPTION:
            return {"input_ids": None, "labels": None}
        # TODO: currently no data augmentation implemented
        # Convert the whole score into chunks
        # TODO: what about conditioning tokens?
        chunked = self.chunker(score)
        # Get the chunk we desire
        desired_chunk = chunked[chunk_idx]
        # Pad the chunk if required
        if len(desired_chunk) < self.max_seq_len + 1:
            desired_chunk = pad_sequence(desired_chunk, self.max_seq_len + 1, self.tokenizer.pad_token_id)
        # Shift to get the target labels for autoregression
        targets = desired_chunk[1:]
        # Remove the last token from the desired sequence
        input_ids = desired_chunk[:-1]
        assert len(targets) == len(input_ids) == self.max_seq_len
        # Assemble everything into the dictionary format
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(targets, dtype=torch.long)
        }

    def __len__(self):
        return len(self.chunk_paths_and_idxs)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"Exhaustive dataloader with {len(self.files_paths)} files, {len(self.chunk_paths_and_idxs)} chunks."


if __name__ == "__main__":
    from miditok import REMI

    # Get a tokenizer with default arguments
    token_factory = REMI()
    # Get filepaths for all MIDI files in the /data/raw/ directories
    midi_paths = utils.get_data_files_with_ext(ext="**/*.mid")[:10]

    # Test out our random chunking dataloader
    dm = DatasetMIDIRandomChunk(
        token_factory,
        midi_paths,
        max_seq_len=100,
        do_augmentation=True
    )
    print(dm)
    for i in range(10):
        item = dm.__getitem__(i)
        print(f'Item {i}, inputs shape {item["input_ids"].size(0)}, labels shape {item["labels"].size(0)}')

    # Test out our exhaustive dataloader
    dm = DatasetMIDIExhaustive(
        token_factory,
        midi_paths,
        max_seq_len=100,
        do_augmentation=False
    )
    print(dm)
    for i in range(10):
        item = dm.__getitem__(i)
        print(f'Item {i}, inputs shape {item["input_ids"].size(0)}, labels shape {item["labels"].size(0)}')
