#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Data loader and collator modules"""

import os
import random
from copy import deepcopy
from time import time

import numpy as np
import torch
from symusic import Score
from tqdm import tqdm

from jazz_style_conditioned_generation import utils
from jazz_style_conditioned_generation.data.augmentation import data_augmentation
from jazz_style_conditioned_generation.data.conditions import (
    get_pianist_tokens,
    get_genre_tokens,
    get_tempo_token,
    get_time_signature_token,
)
from jazz_style_conditioned_generation.data.scores import (
    load_score,
    preprocess_score
)

__all__ = [
    "DATA_DIR",
    "pad_sequence",
    "create_padding_mask",
    "DatasetMIDIExhaustive",
    "DatasetMIDIRandomChunk"
]

DATA_DIR = os.path.join(utils.get_project_root(), "data")


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


def create_padding_mask(x, pad_token_id: int) -> torch.tensor:
    """Create masking tensor that gives 0 when token is pad_token_id, 1 elsewhere"""
    # NB. be aware that causal masks are handled by models: this mask is for padding only
    #  This is identical to the approach in miditok.pytorch_data.DataCollator
    if isinstance(x, list):
        x = torch.tensor(x, dtype=torch.long)
    # Should be True if padding, False otherwise
    return x == pad_token_id


class DatasetMIDIConditioned:
    """Parent dataset class"""
    def __init__(
            self,
            tokenizer,
            files_paths: list[str],
            max_seq_len: int,
            do_augmentation: bool = False,
            do_conditioning: bool = True,
            n_clips: int = None,
    ):
        # Set attributes
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.do_augmentation = do_augmentation
        self.do_conditioning = do_conditioning

        # MIDI file paths
        self.files_paths = files_paths
        utils.validate_paths(self.files_paths, expected_extension=".mid")
        if n_clips is not None:
            self.files_paths = self.files_paths[:n_clips]
            random.shuffle(self.files_paths)

    def add_beginning_and_ending_tokens_to_sequence(
            self,
            tokseq_ids: list[int],
    ) -> list[int]:
        """Adds beginning and ending of sequence tokens to a COMPLETE track"""
        # Add BOS and EOS tokens in
        tokseq_ids.insert(0, self.tokenizer["BOS_None"])
        tokseq_ids.insert(len(tokseq_ids), self.tokenizer["EOS_None"])
        # Sanity check everything is in its right place
        assert tokseq_ids[0] == self.tokenizer["BOS_None"]
        assert tokseq_ids[-1] == self.tokenizer["EOS_None"]
        return tokseq_ids

    def get_conditioning_tokens(self, metadata_fpath: str, tempo_mod: float = 1.):
        """Get conditioning tokens from a metadata JSON. tempo_mod modifies retrieved tempo in case of augmentation"""
        # Read the metadata JSON file (with a large cache to prevent redundant reads)
        metadata_read = utils.read_json_cached(metadata_fpath)
        # Grab the condition tokens for this track (genre, pianist)
        extra_tokens = [
            *get_genre_tokens(metadata_read, self.tokenizer, n_genres=None),  # use all genres
            *get_pianist_tokens(metadata_read, self.tokenizer, n_pianists=1)  # only track pianist
        ]
        # Also grab the tempo and time signature tokens, if we have them
        if "tempo" in metadata_read.keys():
            tempo = metadata_read["tempo"] * tempo_mod  # accounts for data augmentation
            extra_tokens.append(get_tempo_token(tempo, self.tokenizer))
        if "time_signature" in metadata_read.keys():
            extra_tokens.append(get_time_signature_token(metadata_read["time_signature"], self.tokenizer))
        # Convert the extra tokens into token indices
        extra_token_idxs = [self.tokenizer[et] for et in extra_tokens]
        # Sanity checking that there are no duplicate tokens
        assert len(set(extra_token_idxs)) == len(extra_token_idxs), "Duplicates found in conditioning tokens!"
        return extra_token_idxs

    def add_condition_tokens_to_input(
            self,
            sequence: list[int],
            conditions: list[int],
    ) -> tuple[list[int], list[int]]:
        """Add condition tokens to a sequence, preserving the target length"""
        assert len(conditions) > 0, "Condition token list is empty"
        assert len(sequence) == self.max_seq_len, "Got fewer tokens than expected in input sequence!"
        # Condition tokens go before the beginning of the sequence
        comb = conditions + sequence
        # Chunk everything to the required length and sanity check
        x = comb[:self.max_seq_len]
        targets = comb[1: self.max_seq_len + 1]
        assert len(x) == len(targets) == len(sequence) == self.max_seq_len
        return x, targets
        # NB: to "see" the raw tokens from input_ids (i.e., removing any BPE), we can do the following
        # tokens = self.tokenizer._convert_sequence_to_tokseq(torch.tensor([input_ids]))
        # self.tokenizer._preprocess_tokseq_before_decoding(tokens[0])
        # print(tokens[0].tokens)

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self):
        return len(self.files_paths)


class DatasetMIDIRandomChunk(DatasetMIDIConditioned):
    """Dataloader that returns a random chunk of `max_seq_len` for a single track"""

    def __init__(
            self,
            tokenizer,
            files_paths: list[str],
            max_seq_len: int,
            do_augmentation: bool = True,
            do_conditioning: bool = True,
            n_clips: int = None,
            chunk_end_overlap: float = 0.5,
    ):
        super().__init__(tokenizer, files_paths, max_seq_len, do_augmentation, do_conditioning, n_clips)
        self.chunk_end_overlap = chunk_end_overlap
        # Get metadata paths if required, used for conditioning
        if self.do_conditioning:
            self.metadata_paths = [
                fp.replace("piano_midi.mid", "metadata_tivo.json") for fp in self.files_paths
            ]
            utils.validate_paths(self.metadata_paths, expected_extension=".json")

    def randomly_slice_sequence(
            self,
            sequence: list[int],
    ) -> list[int]:
        """Randomly slices a sequence into desired length, allowing for some overlap with the end of the sequence"""
        assert len(sequence) >= self.max_seq_len, f"Expected {self.max_seq_len} tokens but got {len(sequence)}!"
        # end_overlap is the fraction of the total sequence length that we should allow to possibly overlap at the end.
        #  so, with a sequence of length 100 and a desired_len of 50, an end_overlap of 0.5 would mean that we can
        #  possibly create chunks that have indices 75 and 125. This will result in shorter-than-desired sequences,
        #  so must be padded. With an end_overlap of 0, chunks will never overlap the end of the sequence
        #  and will always be the desired length
        assert 0. <= self.chunk_end_overlap < 1., (f'`chunk_end_overlap` must be in range 0. <= x < 1. '
                                                   f'but got {self.chunk_end_overlap}')
        end_range = int(len(sequence) - (self.max_seq_len * (1 - self.chunk_end_overlap)))
        start = random.randint(0, end_range)
        end = start + self.max_seq_len
        return sequence[start:end]

    def chunk_sequence(self, sequence: list[int]) -> tuple[list[int], list[int]]:
        # We need an overlap of 1 to shift the targets for autoregressive modelling
        full_seq_length = self.max_seq_len + 1
        # If the full sequence is too short
        if len(sequence) < full_seq_length:
            # We can just pad the sequence
            x = pad_sequence(sequence, desired_len=full_seq_length, pad_token_id=self.tokenizer.pad_token_id)
        # Randomly chunking the sequence
        else:
            x = self.randomly_slice_sequence(sequence)
            if len(x) < full_seq_length:
                x = pad_sequence(x, desired_len=full_seq_length, pad_token_id=self.tokenizer.pad_token_id)
        # Shift labels by one for autoregression
        targets = x[1:]
        x = x[:-1]  # remove the final token to get the desired length
        assert len(targets) == len(x) == self.max_seq_len
        return x, targets

    def load_file(self, filepath: str) -> tuple[torch.LongTensor, torch.LongTensor, float]:
        # Convert filepath into a symusic object with our custom loading function
        # Previously we were catching SCORE_LOADING_EXCEPTION here silently, but I don't think we want to do this
        score = load_score(filepath)
        # Apply our own preprocessing to the score
        preprocessed_score = preprocess_score(score)
        # Perform data augmentation on the score object if required
        if self.do_augmentation:
            preprocessed_score, tempo_scale = data_augmentation(preprocessed_score)
        else:
            tempo_scale = 1.
        # Now we can do the MIDItok preprocessing
        before_tokenize = preprocess_score(preprocessed_score)
        # Tokenize it
        # tokseq = self.tokenizer.encode(before_tokenize, no_preprocess_score=True)
        tokseq = self.tokenizer.encode(before_tokenize)
        # Add BOS and EOS tokens in
        # MIDITok only adds a BOS and EOS token at the beginning and ending of the track
        temp_ids = deepcopy(tokseq[0].ids)
        # TODO: confirm that we don't need BOS and EOS tokens at the end of EVERY sequence
        tokseq_ids = self.add_beginning_and_ending_tokens_to_sequence(temp_ids)
        # Get a random chunk from the sequence
        input_ids, targets = self.chunk_sequence(tokseq_ids)
        return input_ids, targets, tempo_scale

    def __getitem__(self, idx: int) -> dict[str, torch.LongTensor]:
        # Get the filepath for the corresponding track
        fp = self.files_paths[idx]
        # Load the file and convert to token indices
        input_ids, targets, tempo_scale = self.load_file(fp)
        # Add the conditioning tokens in to the start of the sequence, if required
        if self.do_conditioning:
            condition_idxs = self.get_conditioning_tokens(self.metadata_paths[idx], tempo_scale)
            assert len([ex for ex in condition_idxs if ex in input_ids]) == 0, "Conditioning tokens found in input!"
            # If we actually have condition tokens
            if len(condition_idxs) > 0:
                # Add them to the sequence, preserving the target length and the BOS token (if this is present)
                input_ids, targets = self.add_condition_tokens_to_input(sequence=input_ids, conditions=condition_idxs)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(targets, dtype=torch.long),
            # Mask is for padding only: causal mask is handled by models
            "attention_mask": create_padding_mask(input_ids, self.tokenizer.pad_token_id)
        }

    def __str__(self) -> str:
        return f"Random chunk dataloader with {len(self.files_paths)} tracks/chunks."


class DatasetMIDIExhaustive(DatasetMIDIConditioned):
    """Dataset that returns all `max_seq_len` chunks from a single track"""

    def __init__(
            self,
            tokenizer,
            files_paths: list[str],
            max_seq_len: int,
            do_augmentation: bool = False,
            do_conditioning: bool = True,
            n_clips: int = None,
    ):
        super().__init__(tokenizer, files_paths, max_seq_len, do_augmentation, do_conditioning, n_clips)
        if self.do_augmentation:
            raise NotImplementedError("Data augmentation not implemented for exhaustive MIDI loader")

        # [(filename1, chunk1), (filename1, chunk2), (filename2, chunk1), ...]
        self.chunk_paths_and_idxs = list(self.get_chunks_per_track())

        # Get metadata paths if required
        if self.do_conditioning:
            self.metadata_paths = [
                fp.replace("piano_midi.mid", "metadata_tivo.json")
                for fp, _ in self.chunk_paths_and_idxs
            ]
            utils.validate_paths(self.metadata_paths, expected_extension=".json")
            assert len(self.metadata_paths) == len(self.chunk_paths_and_idxs)

    def chunker(self, score: Score) -> list[list[int]]:
        """Chunks a symusic.Score object into chunks of `max_seq_len` size"""
        # Preprocess the music file with the tokenizer
        score = preprocess_score(score)
        # Tokenize it
        # tokseq = self.tokenizer.encode(score, no_preprocess_score=True)
        tokseq = self.tokenizer.encode(score)
        temp_ids = deepcopy(tokseq[0].ids)
        # Add beginning and ending of sequence tokens in
        tokseq = self.add_beginning_and_ending_tokens_to_sequence(temp_ids)
        # Split into chunks of `max_seq_len` size, plus 1 for autoregressive label shifting
        return [t.tolist() for t in torch.split(torch.tensor(tokseq), self.max_seq_len + 1)]

    def get_chunks_per_track(self) -> tuple[str, int]:
        """For every track in the list of file paths, gets the number of chunks"""
        # Iterate over every file
        for file in tqdm(self.files_paths, desc='Getting track chunks...'):
            # Open file as a symusic score object
            score = load_score(file)
            # Apply our own preprocessing to the score
            preprocessed_score = preprocess_score(score)
            # Convert into chunks
            chunked_file = self.chunker(preprocessed_score)
            # Yield tuples of (filename, chunk_idx)
            for chunk_idx in range(len(chunked_file)):
                yield file, chunk_idx

    def load_file(self, filepath: str, chunk_idx: int) -> tuple[torch.LongTensor, torch.LongTensor]:
        # Convert filepath into a symusic object with our custom loading function
        # Previously we were catching SCORE_LOADING_EXCEPTION here silently, but I don't think we want to do this
        score = load_score(filepath)
        # Apply our own preprocessing to the score
        preprocessed_score = preprocess_score(score)
        # Convert the whole score into chunks
        chunked = self.chunker(preprocessed_score)
        # Get the chunk we desire
        desired_chunk = chunked[chunk_idx]
        # Pad the chunk if required
        if len(desired_chunk) < self.max_seq_len + 1:
            desired_chunk = pad_sequence(desired_chunk, self.max_seq_len + 1, self.tokenizer.pad_token_id)
        # Shift to get the target labels for autoregression
        input_ids, targets = desired_chunk[:-1], desired_chunk[1:]
        # Sanity checking
        assert len(targets) == len(input_ids) == self.max_seq_len
        return input_ids, targets

    def __getitem__(self, idx: int) -> dict[str, torch.LongTensor]:
        # Get the filepath and idx of the desired chunk
        fp, chunk_idx = self.chunk_paths_and_idxs[idx]
        # Load the file and convert to token indices
        input_ids, targets = self.load_file(fp, chunk_idx)
        # Add the conditioning tokens in to the start of the sequence, if required
        if self.do_conditioning:
            # We pass tempo_scale == 1. so that we won't modify the tempo read from the metadata JSON
            extra_token_idxs = self.get_conditioning_tokens(self.metadata_paths[idx], tempo_mod=1.)
            assert len([ex for ex in extra_token_idxs if ex in input_ids]) == 0, "Conditioning tokens found in input!"
            # If we actually have condition tokens
            if len(extra_token_idxs) > 0:
                # Add them to the sequence, preserving the target length and the BOS token (if this is present)
                input_ids, targets = self.add_condition_tokens_to_input(
                    sequence=input_ids,
                    conditions=extra_token_idxs,
                )
        # Assemble everything into the dictionary format
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(targets, dtype=torch.long),
            # Mask is for padding only: causal mask is handled by models
            "attention_mask": create_padding_mask(input_ids, self.tokenizer.pad_token_id)
        }

    def __len__(self):
        return len(self.chunk_paths_and_idxs)

    def __str__(self) -> str:
        return f"Exhaustive dataloader with {len(self.files_paths)} files, {len(self.chunk_paths_and_idxs)} chunks."


if __name__ == "__main__":
    from miditok import MIDILike, TokenizerConfig
    from jazz_style_conditioned_generation.data.tokenizer import (
        DEFAULT_TOKENIZER_CONFIG,
        add_genres_to_vocab,
        add_pianists_to_vocab,
        add_tempos_to_vocab,
        add_timesignatures_to_vocab,
        train_tokenizer
    )

    # Get a tokenizer with default arguments
    token_factory = MIDILike(TokenizerConfig(**DEFAULT_TOKENIZER_CONFIG))
    # Get filepaths for all MIDI files in the /data/raw/ directories
    midi_paths = utils.get_data_files_with_ext(ext="**/*.mid")[:100]
    metadata_paths = [i.replace("piano_midi.mid", "metadata_tivo.json") for i in midi_paths]
    # Add all of our condition tokens to the tokenizer
    add_pianists_to_vocab(token_factory)
    add_genres_to_vocab(token_factory)
    add_tempos_to_vocab(token_factory, (80, 300), 32)
    add_timesignatures_to_vocab(token_factory, [3, 4])
    # Train the tokenizer with BPE
    train_tokenizer(token_factory, vocab_size=1000, model="BPE", files_paths=midi_paths)
    # Test out our random chunking dataloader
    dm = DatasetMIDIRandomChunk(
        token_factory,
        midi_paths,
        max_seq_len=2048,
        do_augmentation=True,
        do_conditioning=True,
    )
    print(dm)

    all_times = []
    for i in range(len(midi_paths)):
        starter = time()
        item = dm.__getitem__(i)
        all_times.append(time() - starter)
        print(f'Item {i}, inputs shape {item["input_ids"].size(0)}, labels shape {item["labels"].size(0)}')
    print(np.mean(all_times))

    # Test out our exhaustive dataloader
    dm = DatasetMIDIExhaustive(
        token_factory,
        midi_paths,
        max_seq_len=100,
        do_augmentation=False,
        do_conditioning=True,
    )
    print(dm)
    for i in range(len(dm)):
        item = dm.__getitem__(i)
        print(f'Item {i}, inputs shape {item["input_ids"].size(0)}, labels shape {item["labels"].size(0)}')
