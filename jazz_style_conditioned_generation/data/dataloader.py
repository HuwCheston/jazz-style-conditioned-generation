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
    "create_padding_mask",
    "DatasetMIDIConditioned"
]

DATA_DIR = os.path.join(utils.get_project_root(), "data")


def create_padding_mask(x, pad_token_id: int) -> torch.tensor:
    """Create masking tensor that gives 0 when token is pad_token_id, 1 elsewhere"""
    # NB. be aware that causal masks are handled by models: this mask is for padding only
    #  This is identical to the approach in miditok.pytorch_data.DataCollator
    if isinstance(x, list):
        x = torch.tensor(x, dtype=torch.long)
    # Should be True if padding, False otherwise
    return x == pad_token_id


class DatasetMIDIConditioned:
    """Dataset class: slices a track into N MAX_SEQ_LEN chunks, applies augmentation and conditioning"""

    def __init__(
            self,
            tokenizer,
            files_paths: list[str],
            max_seq_len: int,
            do_augmentation: bool = False,
            do_conditioning: bool = True,
            n_clips: int = None,
            overlap: int = None,
            min_seq_len: int = None,
    ):
        # Set attributes
        self.tokenizer = tokenizer
        self.do_augmentation = do_augmentation
        self.do_conditioning = do_conditioning

        # The size of the maximum sequence
        self.max_seq_len = max_seq_len
        # The overlap between successive chunks from the same track (in tokens)
        self.overlap = overlap if overlap is not None else max_seq_len // 2  # default to 50% overlap between chunks
        # The size of the smallest sequence we'll consider during training
        self.min_seq_len = min_seq_len if min_seq_len is not None else max_seq_len // 10  # default to 10% of max
        assert self.min_seq_len <= self.overlap <= self.max_seq_len

        # MIDI file paths
        self.files_paths = files_paths
        utils.validate_paths(self.files_paths, expected_extension=".mid")

        # Metadata file paths
        self.metadata_paths = [fp.replace("piano_midi.mid", "metadata_tivo.json") for fp in self.files_paths]
        utils.validate_paths(self.metadata_paths, expected_extension=".json")

        # Preloaded tuples of (score, (seq_start, seq_end), metadata): can have many items for 1 track in the dataset
        self.track_slices = list(self.preload_track_slices())
        if n_clips is not None:
            random.shuffle(self.track_slices)
            self.track_slices = self.track_slices[:n_clips]

    def score_to_token_sequence(self, score: Score, add_bos_eos: bool = True) -> list[int]:
        """Converts a (loaded, preprocessed) score into a token sequence with the tokenizer, also adds BOS/EOS"""
        # Tokenise the score and get the token IDs
        tokenized = self.tokenizer.encode(score)
        tok_ids = tokenized[0].ids
        # Add in the BOS and EOS tokens
        if add_bos_eos:
            tok_ids = self.add_beginning_and_ending_tokens_to_sequence(tok_ids)
        return tok_ids

    def slice_token_sequence(
            self,
            token_sequence: list[int],
            sequence_length: int = None,
            min_sequence_length: int = None,
            overlap: int = None
    ) -> list[tuple[int, int]]:
        """Slices a sequence of tokens into sequence_len chunks (with overlap) and returns starting and stopping idxs"""
        if sequence_length is None:
            sequence_length = self.max_seq_len
        if min_sequence_length is None:
            min_sequence_length = self.min_seq_len
        if overlap is None:
            overlap = self.overlap
        # Split the full score into separate sequences, with our overlap
        all_slice_idxs = []
        for slice_begin in range(0, len(token_sequence), sequence_length - overlap):
            # Break out of the loop once we've finished chunking the track
            if slice_begin >= len(token_sequence):
                break
            # Get the end point of the current slice
            slice_end = slice_begin + sequence_length
            # Slice according to the beginning and ending point
            tokseq_slice = token_sequence[slice_begin:slice_end]
            # This drops very small slices
            if len(tokseq_slice) < min_sequence_length:
                break
            assert min_sequence_length <= len(tokseq_slice) <= sequence_length
            # Append the INDICES that this slice begins/ends at: we may need to modify these after augmenting!
            all_slice_idxs.append((slice_begin, slice_end))
        return all_slice_idxs

    def preload_track_slices(self) -> tuple[Score, [int, int], dict]:
        """For every track, return tuple of FULL SCORE, (slice_begin, slice_end), METADATA"""
        # Iterate over every file
        for midi_file, json_file in tqdm(
                zip(self.files_paths, self.metadata_paths),
                desc='Getting track chunks...',
                total=len(self.files_paths)
        ):
            # If we're doing conditioning, we need to know the number of condition tokens we'll be adding to this track
            if self.do_conditioning:
                # Load up the metadata for this file
                metadata = utils.read_json_cached(json_file)
                # Get the number of conditioning tokens: this will affect our sequence length
                n_condition_tokens = len(self.get_conditioning_tokens(metadata))
            else:
                metadata = dict()
                n_condition_tokens = 0

            # Open MIDI file as a symusic score object
            score = load_score(midi_file)
            # Apply our own preprocessing to the score
            preprocessed_score = preprocess_score(score)
            # Tokenise the score and get the token IDs
            ids_with_bos_eos = self.score_to_token_sequence(preprocessed_score, add_bos_eos=True)

            # Our sequence length depends on the number of conditioning tokens, which we'll add inside __getitem__
            #  i.e., if we want sequences of length 100, but this track has 10 condition tokens, we actually want
            #  to instead get sequences of length 90, so that we can add our condition tokens in later without missing
            #  anything out of the original sequence
            sequence_length = self.max_seq_len - n_condition_tokens
            all_slices_idxs = self.slice_token_sequence(ids_with_bos_eos, sequence_length)

            # For every track, we return the FULL SCORE, the slice indices, and the metadata JSON file
            for slice_begin, slice_end in all_slices_idxs:
                metadata_copy = deepcopy(metadata)  # we need to make a copy so we don't modify the underlying object
                yield preprocessed_score, (slice_begin, slice_end), metadata_copy

    def add_beginning_and_ending_tokens_to_sequence(self, token_ids: list[int]) -> list[int]:
        """Adds beginning and ending of sequence tokens to a COMPLETE track"""
        # Make a deepcopy so we don't modify the underlying list
        temp_ids = deepcopy(token_ids)
        # Add BOS and EOS tokens in
        temp_ids.insert(0, self.tokenizer["BOS_None"])
        temp_ids.insert(len(temp_ids), self.tokenizer["EOS_None"])
        # Sanity check everything is in its right place and that the list is the correct length
        assert temp_ids[0] == self.tokenizer["BOS_None"]
        assert temp_ids[-1] == self.tokenizer["EOS_None"]
        assert len(temp_ids) == len(token_ids) + 2  # we should have only added two tokens to the sequence
        return temp_ids

    def get_conditioning_tokens(self, metadata: dict):
        """Get conditioning tokens from a metadata JSON."""
        # Grab the condition tokens for this track (genre, pianist)
        extra_tokens = [
            *get_genre_tokens(metadata, self.tokenizer),
            *get_pianist_tokens(metadata, self.tokenizer)
        ]
        # Also grab the tempo and time signature tokens, if we have them
        if "tempo" in metadata.keys():
            extra_tokens.append(get_tempo_token(metadata["tempo"], self.tokenizer))
        if "time_signature" in metadata.keys():
            extra_tokens.append(get_time_signature_token(metadata["time_signature"], self.tokenizer))
        # Convert the extra tokens into token indices
        extra_token_idxs = [self.tokenizer[et] for et in extra_tokens]
        # Sanity checking that there are no duplicate tokens
        assert len(set(extra_token_idxs)) == len(extra_token_idxs), "Duplicates found in conditioning tokens!"
        return extra_token_idxs

    def shift_labels(self, token_sequence: list[int]) -> list[int]:
        """Shifts labels for a sequence "one to the left" to create input and target IDs"""
        assert len(token_sequence) == self.max_seq_len + 1, "Got fewer tokens than expected when creating targets"
        targets = token_sequence[1: self.max_seq_len + 1]
        x = token_sequence[: self.max_seq_len]
        assert len(targets) == len(x) == self.max_seq_len
        return x, targets

    @staticmethod
    def scale_tempo(tempo: float, scale: float) -> float:
        """Scales tempo values in BPM depending on data augmentation"""
        if scale == 1.:
            return tempo  # should prevent any rounding issues
        new_tempo = tempo / scale
        # If track gets longer, BPM should decrease (slower)
        if scale > 1.:
            assert new_tempo < tempo
        # If track gets shorter, BPM should increase (faster)
        elif scale < 1.:
            assert new_tempo > tempo
        return new_tempo

    @staticmethod
    def scale_slice_indices(slice_start: int, slice_end: int, scale: float) -> tuple[int, int]:
        """Adjust slice start/ending time depending on augmentation"""
        #  i.e., if we're shortening the track, the starting point must also shift earlier
        slice_len = slice_end - slice_start  # i.e., max_seq_len - n_condition_tokens_for_this_track
        slice_start = round(slice_start * scale)
        slice_end = slice_start + slice_len  # this just ensures we're considering the same number of tokens
        return slice_start, slice_end

    def __str__(self) -> str:
        return f"Dataset {len(self.files_paths)} tracks corresponding to {len(self)} slices."

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self):
        return len(self.track_slices)

    def __getitem__(self, idx: int) -> dict[str, torch.LongTensor]:
        # Unpack everything that we've preloaded from our list of tuples
        full_score, (slice_start, slice_end), metadata = self.track_slices[idx]
        # The score is already loaded + preprocessed, so we don't need to call `load_score` + `preprocess_score` here

        # Perform data augmentation on the score object if required
        if self.do_augmentation:
            full_score, tempo_scale = data_augmentation(full_score)

            # Scale the slice start and ending points by the tempo modulation so we start in the correct place
            slice_start, slice_end = self.scale_slice_indices(slice_start, slice_end, tempo_scale)

            # Adjust track tempo in metadata if required
            if "tempo" in metadata.keys():
                # If we didn't make a copy of this object earlier, this line would modify the metadata object
                #  FOR ALL SLICES of the same underlying track!
                metadata["tempo"] = self.scale_tempo(metadata["tempo"], tempo_scale)

        # Tokenise the score (with BOS + EOS tokens) and get the IDs
        tokseq_ids = self.score_to_token_sequence(full_score, add_bos_eos=True)

        # Now, we can truncate the score according to our slice starting + stopping points
        #  We add one so that we have enough tokens for autoregressive label shifting later on
        tokseq_ids_chunked = tokseq_ids[slice_start: slice_end + 1]

        # If we're conditioning, we need to add the condition tokens to the token sequence
        if self.do_conditioning:
            condition_tokens = self.get_conditioning_tokens(metadata)
            # Sanity checks
            # Condition tokens + sliced sequence length == maximum sequence we want to consider
            assert len(condition_tokens) + (slice_end - slice_start) == self.max_seq_len
            # No conditioning tokens should be in the input sequence (and vice versa)
            assert not set(condition_tokens) & set(tokseq_ids_chunked)
            # Combine everything into a single list of integers, with conditioning tokens at the start
            tokseq_ids_chunked = condition_tokens + tokseq_ids_chunked  # type: list[int]
        # Otherwise, set this to an empty list
        else:
            condition_tokens = []

        # Before padding/truncating, sanity check that we have enough tokens in the sequence
        # TODO: this is flaky? think more about min_seq_len in general
        # assert len(tokseq_ids_chunked) >= self.min_seq_len

        # Pad or truncate the sequence if required
        #  Again, add one to the maximum sequence length so that we have enough tokens for autoregressive shifting later
        if len(tokseq_ids_chunked) < self.max_seq_len + 1:
            tokseq_ids_chunked = utils.pad_sequence(
                tokseq_ids_chunked, desired_len=self.max_seq_len + 1, pad_token_id=self.tokenizer["PAD_None"]
            )
        else:
            tokseq_ids_chunked = tokseq_ids_chunked[: self.max_seq_len + 1]
        assert len(tokseq_ids_chunked) == self.max_seq_len + 1

        # Shift labels for autoregressive teacher forcing
        input_ids, targets = self.shift_labels(tokseq_ids_chunked)
        # Now, everything should be equivalent to the desired sequence length
        assert len(input_ids) == len(targets) == self.max_seq_len

        # Return everything nicely formatted as a dictionary
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(targets, dtype=torch.long),
            # Mask is for padding only: causal mask is handled by models
            "attention_mask": create_padding_mask(input_ids, self.tokenizer.pad_token_id),
            # TODO: maybe we want to append these in a data collator (for different types of conditioning)
            # We have to pad the condition IDs or else we get an error when creating the dataloader
            "condition_ids": torch.tensor(
                utils.pad_sequence(condition_tokens, len(input_ids), self.tokenizer.pad_token_id), dtype=torch.long
            ),
        }


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
    dm = DatasetMIDIConditioned(
        token_factory,
        midi_paths,
        max_seq_len=2048,
        do_augmentation=True,
        do_conditioning=True,
    )
    dm = torch.utils.data.DataLoader(dm, batch_size=2, shuffle=True, drop_last=False)
    print(dm)

    all_times = []
    for i in dm:
        starter = time()
        item = dm.dataset.__getitem__(i)
        all_times.append(time() - starter)
        print(f'Item {i}, inputs shape {item["input_ids"].size(0)}, labels shape {item["labels"].size(0)}')
    print(np.mean(all_times))
