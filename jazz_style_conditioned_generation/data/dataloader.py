#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data loader and collator modules.

There are three data loaders contained here:
- DatasetMIDIConditionedNoOverlapChunks:
    This returns every possible chunk of N tokens from a single track, with no overlap between consecutive chunks.
    So, for a track with 500 tokens, we would expect tokens (0, 99), (100, 199), (200, 299), (300, 399), (400, 499)...
    Chunks smaller than 10% the desired sequence length are dropped. This dataset is used during validation and
    testing: note that, during pretraining, there is no testing stage.

- DatasetMIDIConditionedRandomChunk
    This returns a single random chunk from a track, with a different chunk sampled every epoch. Chunks are sampled
    from the beginning 90% of a track to ensure that they are not too small (e.g., not starting on the last token).
    So, on epoch 1 for a track with 500 tokens, we might get (55, 154), then epoch 2 we get (350, 449), etc.
    This dataset is used during training.

- DatasetMIDIConditionedFullTrack
    This returns an entire track of N tokens, where N may be larger or smaller than the transformer maximum sequence
    length. We then process this in MusicTransformer.evaluate using a sliding window, where we go e.g. from tokens
    (0, 99), (1, 100), (2, 101), (3, 102), etc., in order to get an estimate of the loss for every single token
    in every single track. This dataset is currently not used.

"""

import os
import random
from copy import deepcopy
from time import time

import numpy as np
import torch
from symusic import Score
from tqdm import tqdm

from jazz_style_conditioned_generation import utils
from jazz_style_conditioned_generation.data.augmentation import (
    data_augmentation,
    DURATION_AUGMENT_RANGE,
    PITCH_AUGMENT_RANGE,
    VELOCITY_AUGMENT_RANGE
)
from jazz_style_conditioned_generation.data.conditions import (
    get_pianist_tokens,
    get_genre_tokens,
    get_tempo_token,
    get_time_signature_token,
    get_recording_year_token,
    MAX_GENRE_TOKENS_PER_TRACK,
    MAX_PIANIST_TOKENS_PER_TRACK
)
from jazz_style_conditioned_generation.data.scores import (
    load_score,
    preprocess_score
)

__all__ = [
    "DATA_DIR",
    "create_padding_mask",
    "DatasetMIDIConditionedNoOverlapChunks",
    "DatasetMIDIConditionedRandomChunk",
    "DatasetMIDIConditionedFullTrack"
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


class DatasetMIDIConditionedNoOverlapChunks:
    """Dataset class: slices a track into N MAX_SEQ_LEN chunks (no overlap), applies augmentation and conditioning"""

    def __init__(
            self,
            tokenizer,
            files_paths: list[str],
            max_seq_len: int,
            do_augmentation: bool = False,
            do_conditioning: bool = True,
            n_clips: int = None,
            max_pianist_tokens: int = MAX_PIANIST_TOKENS_PER_TRACK,
            max_genre_tokens: int = MAX_GENRE_TOKENS_PER_TRACK,
            duration_augment_range: list = DURATION_AUGMENT_RANGE,
            pitch_augment_range: list = PITCH_AUGMENT_RANGE,
            velocity_augment_range: list = VELOCITY_AUGMENT_RANGE
    ):
        # Set attributes
        self.tokenizer = tokenizer
        self._do_augmentation = do_augmentation
        self.do_conditioning = do_conditioning
        self.max_pianist_tokens = max_pianist_tokens
        self.max_genre_tokens = max_genre_tokens
        # Range of values used for velocity augmentation across duration, pitch, and velocity
        self.duration_augment_range = duration_augment_range
        self.pitch_augment_range = pitch_augment_range
        self.velocity_augment_range = velocity_augment_range

        # Whether we're tokenizing as seconds or ticks
        self.scores_are_seconds_ttype = self.tokenizer.__class__.__name__ == "CustomTSD"

        # The size of the maximum sequence
        self.max_seq_len = max_seq_len
        # The size of the smallest sequence we'll consider during training
        self.min_seq_len = max_seq_len // 10  # default to 10% of desired sequence length

        # MIDI file paths
        self.files_paths = files_paths
        utils.validate_paths(self.files_paths, expected_extension=".mid")

        # Metadata file paths
        self.metadata_paths = [fp.replace("piano_midi.mid", "metadata_tivo.json") for fp in self.files_paths]
        if self.do_conditioning:  # files only need to exist when we're actually going to use them...
            utils.validate_paths(self.metadata_paths, expected_extension=".json")

        # Preloaded tuples of (score, (seq_start, seq_end), metadata): can have many items for 1 track in the dataset
        self.preloaded_data = list(self.preload_data())
        if n_clips is not None:
            random.shuffle(self.preloaded_data)
            self.preloaded_data = self.preloaded_data[:n_clips]

    @property
    def do_augmentation(self) -> bool:
        """We do not allow augmentation for this class as it would slow down our __getitem__ calls and isn't used"""
        # This method should be overridden in any class that *does* allow for augmentation
        if self._do_augmentation:
            raise NotImplementedError(f"Augmentation not implemented for class {self.__class__.__name__}")
        else:
            return self._do_augmentation

    def score_to_token_sequence(self, score: Score, add_bos_eos: bool = True) -> list[int]:
        """Converts a (loaded, preprocessed) score into a token sequence with the tokenizer, also adds BOS/EOS"""
        # Tokenise the score and get the token IDs
        tokenized = self.tokenizer.encode(score)
        tok_ids = tokenized[0].ids
        # Add in the BOS and EOS tokens
        if add_bos_eos:
            tok_ids = self.add_beginning_and_ending_tokens_to_sequence(tok_ids)
        return tok_ids

    def preload_data(self) -> tuple:
        """For every track, return the tokenized score as chunks of max_seq_len, with condition tokens"""
        # We don't allow for augmentation in this class
        assert self.do_augmentation is False
        # Iterate over every file
        for midi_file, json_file in tqdm(
                zip(self.files_paths, self.metadata_paths),
                desc='Getting track chunks (exhaustive)...',
                total=len(self.files_paths)
        ):
            # Load up the metadata for this file and get the conditioning tokens
            if self.do_conditioning:
                metadata = utils.read_json_cached(json_file)
                condition_tokens = self.get_conditioning_tokens(metadata)
            else:
                condition_tokens = []

            # Open MIDI file as a symusic score object
            score = load_score(midi_file, as_seconds=self.scores_are_seconds_ttype)
            # Apply our own preprocessing to the score
            preprocessed_score = preprocess_score(score)
            # Tokenise the score and get the token IDs
            ids_with_bos_eos = self.score_to_token_sequence(preprocessed_score, add_bos_eos=True)
            # No conditioning tokens should be in the input sequence (and vice versa)
            assert not set(condition_tokens) & set(ids_with_bos_eos)
            # Our sequence length depends on the number of conditioning tokens
            #  i.e., if we want sequences of length 100, but this track has 10 condition tokens, we actually want
            #  to instead get sequences of length 90, so that we can add our condition tokens in later without missing
            #  anything out of the original sequence
            sequence_length = self.max_seq_len - len(condition_tokens)
            all_slices_idxs = [
                # Start, end of every slice
                (i, i + sequence_length)
                for i in range(0, len(ids_with_bos_eos), sequence_length)
                # This drops sequences that are too short (i.e., those taken from the very end of a track)
                if min(i + sequence_length, len(ids_with_bos_eos)) - i >= self.min_seq_len
            ]
            # Iterate over all of our slice start and end points
            for slice_start, slice_end in all_slices_idxs:
                # Chunk the token sequence
                # We add one here so that we have enough tokens for autoregressive label shifting inside __getitem__
                tokseq_chunked = ids_with_bos_eos[slice_start:slice_end + 1]
                # Add in the condition tokens (will be empty if no conditioning)
                tokseq_conditioned = condition_tokens + tokseq_chunked
                # Condition tokens + sliced sequence length == maximum sequence we want to consider
                assert len(condition_tokens) + (slice_end - slice_start) == self.max_seq_len
                # If the sequence is too short, pad it (again, add one for autoregressive shifting)
                if len(tokseq_conditioned) < self.max_seq_len + 1:
                    tokseq_final = utils.pad_sequence(
                        tokseq_conditioned,
                        desired_len=self.max_seq_len + 1,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                # Otherwise, if the sequence is too long, truncate it (again, add one for autoregressive shifting)
                else:
                    tokseq_final = tokseq_conditioned[: self.max_seq_len + 1]
                # Length of the sequence should be as desired
                assert len(tokseq_final) == self.max_seq_len + 1
                # Return the tokenized sequence, with our condition tokens
                yield tokseq_final, condition_tokens

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
            *get_genre_tokens(metadata, self.tokenizer, n_genres=self.max_genre_tokens),
            *get_pianist_tokens(metadata, self.tokenizer, n_pianists=self.max_pianist_tokens)
        ]
        # Also grab the tempo, time signature, and year tokens, if we have them
        if "recording_year" in metadata.keys() and metadata["recording_year"] is not None:
            extra_tokens.append(get_recording_year_token(metadata["recording_year"], self.tokenizer))
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
        return f"Dataset with {len(self)} track slices."

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        """Returns the number of preloaded track chunks"""
        return len(self.preloaded_data)

    def __getitem__(self, idx: int) -> dict[str, torch.LongTensor]:
        # We don't allow for augmentation in this class
        assert self.do_augmentation is False
        # Unpack the pre-tokenized chunk of the score + condition tokens
        #  We make a copy here so that we don't modify the underlying object
        loaded = deepcopy(self.preloaded_data[idx])
        # This is our pre-chunked token sequence with the condition tokens added
        #  and also just our condition tokens, without the music tokens
        tokseq_ids_chunked, condition_tokens = loaded
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
            # We have to pad the condition IDs or else we get an error when creating the dataloader
            "condition_ids": torch.tensor(
                utils.pad_sequence(condition_tokens, len(input_ids), self.tokenizer.pad_token_id), dtype=torch.long
            ),
        }


class DatasetMIDIConditionedRandomChunk(DatasetMIDIConditionedNoOverlapChunks):
    """Training dataloader: slices a track into a different random chunk of MAX_SEQ_LEN tokens every epoch"""

    START_TOKENS = ("BOS", "TimeShift", "NoteOn", "Pitch", "Chord", "Bar")

    def __init__(
            self,
            tokenizer,
            files_paths: list[str],
            max_seq_len: int,
            do_augmentation: bool = False,
            do_conditioning: bool = True,
            n_clips: int = None,
            max_pianist_tokens: int = MAX_PIANIST_TOKENS_PER_TRACK,
            max_genre_tokens: int = MAX_GENRE_TOKENS_PER_TRACK,
            duration_augment_range: list = DURATION_AUGMENT_RANGE,
            pitch_augment_range: list = PITCH_AUGMENT_RANGE,
            velocity_augment_range: list = VELOCITY_AUGMENT_RANGE
    ):
        super().__init__(
            tokenizer,
            files_paths,
            max_seq_len,
            do_augmentation,
            do_conditioning,
            n_clips,
            max_pianist_tokens,
            max_genre_tokens,
            duration_augment_range,
            pitch_augment_range,
            velocity_augment_range
        )

    @property
    def do_augmentation(self) -> bool:
        """We do allow augmentation for this class"""
        return self._do_augmentation

    def preload_data(self) -> tuple:
        # Iterate over every file
        for midi_file, json_file in tqdm(
                zip(self.files_paths, self.metadata_paths),
                desc='Getting track chunks (random)...',
                total=len(self.files_paths)
        ):
            # If we're doing conditioning, we want to load the metadata for the track
            if self.do_conditioning:
                # Load up the metadata for this file
                metadata = utils.read_json_cached(json_file)
            else:
                metadata = dict()

            # Open MIDI file as a symusic score object
            score = load_score(midi_file, as_seconds=self.scores_are_seconds_ttype)
            # Apply our own preprocessing to the score
            preprocessed_score = preprocess_score(score)
            # Return the preprocessed score and the metadata
            #  We allow for augmentation in this class, which means much of our processing (e.g., tokenization)
            #  will have to be done on the fly, inside __getitem__.
            yield preprocessed_score, metadata

    def get_slice_start_point(self, tokseq_ids: list[int]) -> int:
        """Our random sequence MUST start with a realistic starting token (i.e., not a NoteOff or Velocity token)"""
        # Ensures slices only start at the first 90% of the track (so we don't slice at e.g., the very last token!)
        trunc = tokseq_ids[:-len(tokseq_ids) // 10]
        # Detokenize the IDs into raw tokens: e.g., NoteOn, NoteOff, TimeShift, Velocity tokens
        #  Needs to be handled slightly differently depending on if the tokenizer is trained or not
        if self.tokenizer.is_trained:
            #  This gives us a list of [[Base token 1, Base token 2], [Base token 2], [Base token 2, Base token 3], ...]
            detokenized = [[self.tokenizer[v] for v in self.tokenizer.bpe_token_mapping[i]] for i in trunc]
            # These are the IDXs of the token ID list that decode to one of our acceptable values
            accept_idxs = [
                idx for idx, tok in enumerate(detokenized)
                if any((t.startswith(self.START_TOKENS) for t in tok))
            ]
        # Approx 3x faster than tapping into bpe_token_mapping for a non-trained tokenizer
        else:
            accept_idxs = [idx for idx, i in enumerate(trunc) if self.tokenizer[i].startswith(self.START_TOKENS)]
        # Make a random choice for the starting token
        start = random.choice(accept_idxs)
        return start

    def __getitem__(self, idx: int) -> dict[str, torch.LongTensor]:
        # Unpack everything that we've preloaded from our list of tuples
        #  We make a copy here so that we don't modify the underlying object when we augment
        loaded = deepcopy(self.preloaded_data[idx])
        full_score, metadata = loaded
        # The score is already loaded + preprocessed, so we don't need to call `load_score` + `preprocess_score` here

        # Perform data augmentation on the score object if required
        if self.do_augmentation:
            full_score, tempo_scale = data_augmentation(
                full_score,
                pitch_augmentation_range=self.pitch_augment_range,
                duration_augmentation_range=self.duration_augment_range,
                velocity_augmentation_range=self.velocity_augment_range
            )
            # No need to adjust any of the slice starting/stopping points as we'll sample these later
            # Adjust track tempo in metadata if required
            if "tempo" in metadata.keys():
                # If we didn't make a copy of this object earlier, this line would modify the metadata object
                #  FOR ALL SLICES of the same underlying track!
                metadata["tempo"] = self.scale_tempo(metadata["tempo"], tempo_scale)

        # Tokenise the score (with BOS + EOS tokens) and get the IDs
        tokseq_ids = self.score_to_token_sequence(full_score, add_bos_eos=True)

        # Get the starting and stopping points for the random slice
        #  The starting point MUST be a timeshift, BOS, or note-on token (or equivalent)
        #  This is so that we don't start learning with e.g. a note-off token when there has been no previous note-on
        slice_start = self.get_slice_start_point(tokseq_ids)
        slice_end = slice_start + self.max_seq_len

        # If we're conditioning, we need to get the conditioning tokens
        if self.do_conditioning:
            condition_tokens = self.get_conditioning_tokens(metadata)
            slice_end -= len(condition_tokens)  # account for number of conditioning tokens in sequence length
            # Condition tokens + sliced sequence length == maximum sequence we want to consider
            assert len(condition_tokens) + (slice_end - slice_start) == self.max_seq_len
        # Otherwise, set the conditioning tokens to an empty list
        else:
            condition_tokens = []

        # Now we truncate according to the slice starting + stopping points
        #  We add one so that we have enough tokens for autoregressive label shifting later on
        tokseq_ids_chunked = tokseq_ids[slice_start: slice_end + 1]
        # No conditioning tokens should be in the input sequence (and vice versa)
        assert len([i for i in tokseq_ids_chunked if i in condition_tokens]) == 0
        # Combine everything into a single list of integers, with conditioning tokens at the start
        tokseq_ids_chunked = condition_tokens + tokseq_ids_chunked  # type: list[int]

        # Pad or truncate the sequence if required
        #  Again, add one to the maximum sequence length so that we have enough tokens for autoregressive shifting later
        if len(tokseq_ids_chunked) < self.max_seq_len + 1:
            tokseq_ids_chunked = utils.pad_sequence(
                tokseq_ids_chunked,
                desired_len=self.max_seq_len + 1,
                pad_token_id=self.tokenizer["PAD_None"]
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
            # We have to pad the condition IDs or else we get an error when creating the dataloader
            "condition_ids": torch.tensor(
                utils.pad_sequence(condition_tokens, len(input_ids), self.tokenizer.pad_token_id), dtype=torch.long
            ),
        }


class DatasetMIDIConditionedFullTrack(DatasetMIDIConditionedNoOverlapChunks):
    """Returns FULL LENGTH tracks. Should be used with batch_size=1 in dataloader as sequence lengths are ragged"""

    def __init__(
            self,
            tokenizer,
            files_paths: list[str],
            max_seq_len: int,
            do_augmentation: bool = False,
            do_conditioning: bool = True,
            n_clips: int = None,
            max_pianist_tokens: int = MAX_PIANIST_TOKENS_PER_TRACK,
            max_genre_tokens: int = MAX_GENRE_TOKENS_PER_TRACK,
            duration_augment_range: list = DURATION_AUGMENT_RANGE,
            pitch_augment_range: list = PITCH_AUGMENT_RANGE,
            velocity_augment_range: list = VELOCITY_AUGMENT_RANGE
    ):
        if do_augmentation:
            raise AttributeError("Cannot use augmentation in a dataset that returns full length tracks")
        super().__init__(
            tokenizer,
            files_paths,
            max_seq_len,
            do_augmentation,
            do_conditioning,
            n_clips,
            max_pianist_tokens,
            max_genre_tokens,
            duration_augment_range,
            pitch_augment_range,
            velocity_augment_range
        )

    def preload_data(self) -> tuple:
        # Iterate over every file
        for midi_file, json_file in tqdm(
                zip(self.files_paths, self.metadata_paths),
                desc='Getting full-length tracks...',
                total=len(self.files_paths)
        ):
            # Load up the metadata for this file and get the conditioning tokens
            if self.do_conditioning:
                metadata = utils.read_json_cached(json_file)
                condition_tokens = self.get_conditioning_tokens(metadata)
            else:
                condition_tokens = []
            # Open MIDI file as a symusic score object
            score = load_score(midi_file, as_seconds=self.scores_are_seconds_ttype)
            # Apply our own preprocessing to the score
            preprocessed_score = preprocess_score(score)
            # Tokenise the score and get the token IDs
            ids_with_bos_eos = self.score_to_token_sequence(preprocessed_score, add_bos_eos=True)
            # No conditioning tokens should be in the input sequence (and vice versa)
            assert not set(condition_tokens) & set(ids_with_bos_eos)
            # Add the condition tokens to the full token sequence
            tokseq_final = condition_tokens + ids_with_bos_eos
            # Only padding is required for this class: we need at least max_seq_len + 1 tokens
            if len(tokseq_final) < self.max_seq_len + 1:
                tokseq_final = utils.pad_sequence(tokseq_final, self.max_seq_len + 1, self.tokenizer.pad_token_id)
            # No need to truncate, we want the full length sequence
            yield tokseq_final, condition_tokens

    def shift_labels(self, token_sequence: list[int]) -> list[int]:
        """Overrides base method to allow for sequences of any length"""
        targets = token_sequence[1:]
        x = token_sequence[:-1]
        assert len(targets) == len(x)
        return x, targets

    def __getitem__(self, idx: int) -> dict[str, torch.LongTensor]:
        # Unpack everything that we've preloaded from our list of tuples
        #  This just gives us our token sequence (a full track) and our condition tokens
        #  We don't have to worry about augmentation, padding, or truncation, as this is already handled for us
        loaded = deepcopy(self.preloaded_data[idx])
        tokseq_ids, condition_tokens = loaded
        # Shift labels for autoregressive teacher forcing
        input_ids, targets = self.shift_labels(tokseq_ids)
        assert len(input_ids) >= self.max_seq_len
        # Return everything nicely formatted as a dictionary
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(targets, dtype=torch.long),
            "attention_mask": create_padding_mask(input_ids, self.tokenizer.pad_token_id),
            # We have to pad the condition IDs or else we get an error when creating the dataloader
            "condition_ids": torch.tensor(
                utils.pad_sequence(condition_tokens, len(input_ids), self.tokenizer.pad_token_id), dtype=torch.long
            ),
        }


if __name__ == "__main__":
    from jazz_style_conditioned_generation.data.tokenizer import (
        add_genres_to_vocab,
        add_pianists_to_vocab,
        add_tempos_to_vocab,
        add_timesignatures_to_vocab,
        add_recording_years_to_vocab,
        load_tokenizer
    )

    # Get a MIDILike tokenizer with default arguments
    token_factory = load_tokenizer(tokenizer_str="custom-tsd")
    # Get filepaths for all MIDI files in the /data/raw/ directories
    midi_paths = utils.get_data_files_with_ext(ext="**/*.mid")
    metadata_paths = [i.replace("piano_midi.mid", "metadata_tivo.json") for i in midi_paths]
    # Add all of our condition tokens to the tokenizer
    add_pianists_to_vocab(token_factory)
    add_genres_to_vocab(token_factory)
    add_recording_years_to_vocab(token_factory)
    add_tempos_to_vocab(token_factory, 80, 30, factor=1.05)
    add_timesignatures_to_vocab(token_factory, [3, 4])
    # Train the tokenizer with BPE
    # train_tokenizer(token_factory, vocab_size=1000, model="BPE", files_paths=midi_paths[:100])
    # Test out our random chunking dataloader
    kwargs = dict(
        tokenizer=token_factory,
        files_paths=midi_paths[:10],
        max_seq_len=1024,
        do_conditioning=True
    )
    for dataset_cls, aug in zip(
            [DatasetMIDIConditionedNoOverlapChunks, DatasetMIDIConditionedRandomChunk],
            [False, True]
    ):
        dm = dataset_cls(**kwargs, do_augmentation=aug)
        print(dm)

        all_times = []
        for i in range(len(dm)):
            starter = time()
            item = dm.__getitem__(i)
            all_times.append(time() - starter)
            print(f'Item {i}, inputs shape {item["input_ids"].size(0)}, labels shape {item["labels"].size(0)}')
        print(np.mean(all_times))
