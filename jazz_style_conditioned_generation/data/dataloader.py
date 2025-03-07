#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Data loader and collator modules"""

import os
import random
from copy import deepcopy
from time import time

import numpy as np
import torch
from miditok.constants import SCORE_LOADING_EXCEPTION
from miditok.data_augmentation import augment_score
from symusic import Score, Track, Note
from tqdm import tqdm

from jazz_style_conditioned_generation import utils
from jazz_style_conditioned_generation.data.conditions import (
    validate_conditions, get_condition_special_tokens, validate_condition_values
)

__all__ = [
    "DATA_DIR",
    "add_beginning_and_ending_tokens_to_sequence",
    "pad_sequence",
    "randomly_slice_sequence",
    "get_pitch_augmentation_value",
    "random_data_augmentation",
    "deterministic_data_augmentation",
    "remove_out_of_range_notes",
    "preprocess_score",
    "note_list_to_score",
    "remove_short_notes",
    "merge_repeated_notes",
    "add_condition_tokens_to_sequence",
    "get_conditions_for_track",
    "create_padding_mask",
    "PITCH_AUGMENT_RANGE",
    "DURATION_AUGMENT_RANGE",
    "DatasetMIDIExhaustive",
    "DatasetMIDIRandomChunk"
]

DATA_DIR = os.path.join(utils.get_project_root(), "data")

# DEFAULT_AUGMENTATION_PROB = 0.5
PITCH_AUGMENT_RANGE = range(-3, 4)  # as in Music Transformer
DURATION_AUGMENT_RANGE = [0.95, 0.975, 1.0, 1.025, 1.05]  # as in Music Transformer

OVERLAP_TICKS = 3  # If two notes with the same pitch have less than this offset-onset time, they will be merged
MIN_DURATION_TICKS = 20  # We remove notes that have a duration of less than this value


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


def deterministic_data_augmentation(
        score: Score,
        pitch_augment_value: int,
        duration_augment_value: float
) -> Score:
    """Applies pitch and duration augmentation with specified values to a Score object"""
    # We can just apply the pitch augmentation directly using the MIDITok function
    augmented = augment_score(score, pitch_offset=pitch_augment_value, augment_copy=True, duration_offset=0., )
    # Sanity check: pitches should be within the range of the piano keyboard
    aug_min, aug_max = utils.get_pitch_range(augmented)
    assert aug_min >= utils.MIDI_OFFSET
    assert aug_max <= utils.MIDI_OFFSET + utils.PIANO_KEYS
    # We need to use the symusic pretty_midi-like function to do duration augmentation
    return augmented.adjust_time(
        [augmented.start(), augmented.end()],
        [augmented.start(), int(augmented.end() * duration_augment_value)],
        inplace=False
    )


def random_data_augmentation(
        score: Score,
        pitch_augmentation_range: list = None,
        duration_augmentation_range: list = None
) -> Score:
    """Applies pitch and duration augmentation to a Score object"""
    if pitch_augmentation_range is None:
        pitch_augmentation_range = PITCH_AUGMENT_RANGE
    if duration_augmentation_range is None:
        duration_augmentation_range = DURATION_AUGMENT_RANGE

    # Get random augmentation value from the ranges provided
    pitch_augment = get_pitch_augmentation_value(score, pitch_augmentation_range)
    duration_augment = np.random.choice(duration_augmentation_range)
    # Apply data augmentation with the randomly selected values
    return deterministic_data_augmentation(score, pitch_augment, duration_augment)


def validate_paths(filepaths: list[str], expected_extension: str = ".mid"):
    """Validates that all paths exist on disk and have an expected extension"""
    for file in filepaths:
        assert os.path.isfile(file), f"File {file} does not exist on the disk!"
        assert file.endswith(expected_extension), f"File {file} does not have expected extension {expected_extension}!"


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
        end_overlap: float = 0.5
) -> list[int]:
    """Randomly slices a sequence into desired length, allowing for some overlap with the end of the sequence"""
    assert len(sequence) >= desired_len, f"Expected at least {desired_len} tokens but got {len(sequence)}!"
    # end_overlap is the fraction of the total sequence length that we should allow to possibly overlap at the end
    #  so, with a sequence of length 100 and a desired_len of 50, an end_overlap of 0.5 would mean that we can possibly
    #  create chunks that have indices 75 and 125. This will result in shorter-than-desired sequences, so must be padded
    #  with an end_overlap of 0, chunks will never overlap the end of the sequence and will always be the desired length
    assert 0. <= end_overlap < 1., f'`end_overlap` must be in range 0. <= x < 1. but got {end_overlap}'
    end_range = int(len(sequence) - (desired_len * (1 - end_overlap)))
    start = random.randint(0, end_range)
    end = start + desired_len
    return sequence[start:end]


def remove_short_notes(note_list: list[Note], min_duration_ticks: int = MIN_DURATION_TICKS) -> list[Note]:
    """Removes symusic.Note objects with a duration of less than MIN_DURATION_TICKS from a list of Note objects"""
    newnotes = []
    for note in note_list:
        # Notes with a duration this short are transcription errors usually
        if note.duration >= min_duration_ticks:
            newnotes.append(note)
    return newnotes


def merge_repeated_notes(note_list: list[Note], overlap_ticks: int = OVERLAP_TICKS) -> list[Note]:
    """Merge successive notes at the same pitch with an offset-onset time < OVERLAP_TICKS to a single, long note"""
    newnotes = []
    # Iterate over all MIDI pitches
    for pitch in range(utils.MIDI_OFFSET, utils.MIDI_OFFSET + utils.PIANO_KEYS + 1):
        # Get the notes played at this pitch
        notes_at_pitch = [note for note in note_list if note.pitch == pitch]
        # If this pitch only appears once
        if len(notes_at_pitch) < 2:
            # We can just use it straight away
            newnotes.extend(notes_at_pitch)
        # Otherwise, if we have multiple appearances of this pitch
        else:
            # Sort notes by onset time
            notes_sorted = sorted(notes_at_pitch, key=lambda x: x.time)
            seen_notes = []  # Tracks already processed notes
            # Iterate over successive pairs of notes (note1, note2), (note2, note3)...
            for note_idx in range(len(notes_sorted) - 1):
                # Unpack to get desired notes
                note1 = notes_sorted[note_idx]
                note2 = notes_sorted[note_idx + 1]
                # Check if this note pair has already been merged and processed
                if note1 in seen_notes:
                    continue
                # Unpack everything
                note1_end = note1.time + note1.duration
                note2_start = note2.time
                overlap = note2_start - note1_end
                # If the overlap between these two notes is short
                if overlap < overlap_ticks:
                    # Combine both notes into a single note
                    newnote = Note(
                        # Just use the onset time of the earliest note
                        time=note1.time,
                        # Combine the durations of both notes + the overlap duration
                        duration=note1.duration + overlap + note2.duration,
                        # Pitch should just be the same
                        pitch=note1.pitch,
                        # Take the midpoint of both velocity values
                        velocity=(note1.velocity + note2.velocity) // 2,
                        ttype=note1.ttype
                    )
                    newnotes.append(newnote)
                    seen_notes.append(note1)  # Mark note1 as processed
                    seen_notes.append(note2)  # Mark note2 as processed
                else:
                    # No overlap, append note1 to the newnotes list
                    newnotes.append(note1)
                    seen_notes.append(note1)  # Mark note1 as processed
            # Ensure the last note is added (it might not be part of any overlap)
            if notes_sorted[-1] not in seen_notes:
                newnotes.append(notes_sorted[-1])
    return newnotes


def remove_out_of_range_notes(note_list: list[Note]) -> list[Note]:
    """Remove notes from a list that are outside the range of the piano keyboard"""
    return [n for n in note_list if utils.MIDI_OFFSET <= n.pitch <= utils.MIDI_OFFSET + utils.PIANO_KEYS]


def note_list_to_score(note_list: list[Note], ticks_per_quarter: int) -> Score:
    """Converts a list of symusic.Note objects to a single symusic.Score"""
    # This API is fairly similar to pretty_midi
    newscore = Score()
    newscore.ticks_per_quarter = ticks_per_quarter
    newscore.tracks = [Track()]
    newscore.tracks[0].notes = note_list
    return newscore


def preprocess_score(
        score: Score,
        min_duration_ticks: int = MIN_DURATION_TICKS,
        overlap_ticks: int = OVERLAP_TICKS
) -> Score:
    """Applies our own preprocessing to a Score object: removes short notes, merges duplicates"""
    # Resample the score if required to the value which is used in JTD + PiJAMA
    if score.ticks_per_quarter != utils.TICKS_PER_QUARTER:
        score = score.resample(utils.TICKS_PER_QUARTER)
    assert score.ticks_per_quarter == utils.TICKS_PER_QUARTER
    # If we somehow have more than one track (occasionally happens in the bushgrafts corpus)
    if len(score.tracks) > 1:
        # Get all the piano tracks
        is_piano = [p for p in score.tracks if p.program == utils.MIDI_PIANO_PROGRAM]
        # Keep the one with the most notes
        desired_track = max(is_piano, key=lambda x: len(x.notes))
        note_list = desired_track.notes
    # Otherwise, we can just grab the track directly
    else:
        note_list = score.tracks[0].notes
    # First, we remove notes that are outside the range of the piano keyboard
    validated_notes = remove_out_of_range_notes(note_list)
    # Then, we remove notes with a very short duration
    no_short_notes = remove_short_notes(validated_notes, min_duration_ticks=min_duration_ticks)
    # Next, we merge successive notes with the same pitch and a very short onset-offset time into the same pitch
    merged_notes = merge_repeated_notes(no_short_notes, overlap_ticks=overlap_ticks)
    # Finally, we convert everything back to a Score object that can be passed to our tokenizer
    return note_list_to_score(merged_notes, score.ticks_per_quarter)


def get_conditions_for_track(
        conditions_and_mapping: dict[str, dict],
        metadata: dict,
        tokenizer
) -> list[int]:
    """Given a mapping {condition: {val1: token1}}, convert track metadata into token format"""
    condition_tokens = []
    # By sorting, we ensure that tokens are always inserted in a consistent order
    conditions = sorted(list(conditions_and_mapping.keys()))
    for condition in conditions:
        mapper = conditions_and_mapping[condition]
        values_for_track = metadata[condition]
        if isinstance(values_for_track, list):
            values_for_track = [c["name"] for c in values_for_track]
        # This merges similar values together, removes invalid values etc.
        validated_condition_values = validate_condition_values(values_for_track, condition)
        # This converts values into their token form
        condition_tokens.extend([mapper[c] for c in validated_condition_values if c in mapper.keys()])
    # Sort the tokens alphabetically and return the indices
    return [tokenizer[c] for c in sorted(condition_tokens)]


def add_condition_tokens_to_sequence(
        sequence: list[int],
        condition_tokens: list[int],
) -> tuple[list[int], list[int]]:
    """Add condition tokens to a sequence, preserving length"""
    assert len(condition_tokens) > 0, "Condition token list is empty"
    max_seq_len = len(sequence)
    # Condition tokens go before the beginning of the sequence
    comb = condition_tokens + sequence
    # Chunk everything to the required length and sanity check
    x = comb[:max_seq_len]
    targets = comb[1: max_seq_len + 1]
    assert len(x) == len(targets) == len(sequence)
    return x, targets


def create_padding_mask(x, pad_token_id: int) -> torch.tensor:
    """Create masking tensor that gives 0 when token is pad_token_id, 1 elsewhere"""
    # NB. be aware that causal masks are handled by models: this mask is for padding only
    #  This is identical to the approach in miditok.pytorch_data.DataCollator
    if isinstance(x, list):
        x = torch.tensor(x, dtype=torch.long)
    # Should be True if padding, False otherwise
    return x == pad_token_id


class DatasetMIDIRandomChunk:
    """Dataloader that returns a random chunk of `max_seq_len` for a single track"""
    def __init__(
            self,
            tokenizer,
            files_paths: list[str],
            max_seq_len: int,
            do_augmentation: bool = True,
            do_conditioning: bool = True,
            condition_mapping: dict[str, dict] = None,
            n_clips: int = None,
            chunk_end_overlap: float = 0.5,
            min_duration_ticks: int = MIN_DURATION_TICKS,
            overlap_ticks: int = OVERLAP_TICKS
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.do_augmentation = do_augmentation
        self.chunk_end_overlap = chunk_end_overlap

        # For preprocessing
        self.min_duration_ticks = min_duration_ticks
        self.overlap_ticks = overlap_ticks

        # MIDI file paths
        self.files_paths = files_paths
        validate_paths(self.files_paths, expected_extension=".mid")
        if n_clips is not None:
            self.files_paths = self.files_paths[:n_clips]
            random.shuffle(self.files_paths)

        # Conditioning
        self.do_conditioning = do_conditioning
        if self.do_conditioning:
            if not condition_mapping:
                raise AttributeError("Passed `do_conditioning == True`, but did not pass `condition_mapping`")
            self.conditions = list(condition_mapping.keys())
            self.condition_mapping = condition_mapping
            validate_conditions(self.conditions)
            self.metadata_paths = [
                fp.replace("piano_midi.mid", "metadata_tivo.json") for fp in self.files_paths
            ]
            validate_paths(self.metadata_paths, expected_extension=".json")

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
            x = randomly_slice_sequence(sequence, desired_len=full_seq_length, end_overlap=self.chunk_end_overlap)
            if len(x) < full_seq_length:
                x = pad_sequence(x, desired_len=full_seq_length, pad_token_id=self.tokenizer.pad_token_id)
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
        # Apply our own preprocessing to the score
        preprocessed_score = preprocess_score(
            score,
            min_duration_ticks=self.min_duration_ticks,
            overlap_ticks=self.overlap_ticks
        )
        # Perform data augmentation on the score object if required
        if self.do_augmentation:
            preprocessed_score = random_data_augmentation(preprocessed_score)
        # Now we can do the MIDItok preprocessing
        before_tokenize = self.tokenizer.preprocess_score(preprocessed_score)
        # Tokenize it
        tokseq = self.tokenizer.encode(before_tokenize, no_preprocess_score=True)
        # Add BOS and EOS tokens in
        # MIDITok only adds a BOS and EOS token at the beginning and ending of the track
        temp_ids = deepcopy(tokseq[0].ids)
        # TODO: confirm that we don't need BOS and EOS tokens at the end of EVERY sequence
        tokseq_ids = add_beginning_and_ending_tokens_to_sequence(
            temp_ids, self.tokenizer["BOS_None"], self.tokenizer["EOS_None"]
        )
        # TODO: potentially remove bar/tempo tokens here (they have no meaning)
        # Get a random chunk from the sequence
        input_ids, targets = self.chunk_sequence(tokseq_ids)
        # Add the conditioning tokens in to the start of the sequence, if required
        if self.do_conditioning:
            # Read the metadata JSON file (with a large cache to prevent redundant reads)
            metadata_read = utils.read_json_cached(self.metadata_paths[idx])
            # Grab the condition tokens for this track
            condition_tokens = get_conditions_for_track(self.condition_mapping, metadata_read, self.tokenizer)
            # If we actually have condition tokens
            if len(condition_tokens) > 0:
                # Add them to the sequence, preserving the target length and the BOS token (if this is present)
                input_ids, targets = add_condition_tokens_to_sequence(
                    sequence=input_ids,
                    condition_tokens=condition_tokens,
                )
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(targets, dtype=torch.long),
            # Mask is for padding only: causal mask is handled by models
            "attention_mask": create_padding_mask(input_ids, self.tokenizer.pad_token_id)
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
            do_conditioning: bool = True,
            condition_mapping: dict[str, dict] = None,
            n_clips: int = None,
            min_duration_ticks: int = MIN_DURATION_TICKS,
            overlap_ticks: int = OVERLAP_TICKS
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        if do_augmentation:
            raise NotImplementedError("Data augmentation not implemented for exhaustive MIDI loader")

        # For preprocessing
        self.min_duration_ticks = min_duration_ticks
        self.overlap_ticks = overlap_ticks

        # Conditioning
        self.do_conditioning = do_conditioning
        if self.do_conditioning and not condition_mapping:
            raise AttributeError("Passed `do_conditioning == True`, but no conditions were passed")
        if self.do_conditioning:
            self.conditions = list(condition_mapping.keys())
            self.condition_mapping = condition_mapping
            validate_conditions(self.conditions)

        validate_paths(files_paths, expected_extension=".mid")
        # [filename1, filename2, ...]
        self.files_paths = files_paths
        if n_clips is not None:
            self.files_paths = self.files_paths[:n_clips]
            random.shuffle(self.files_paths)
        # [(filename1, chunk1), (filename1, chunk2), (filename2, chunk1), ...]
        self.chunk_paths_and_idxs = list(self.get_chunks_per_track())

        if self.do_conditioning:
            self.metadata_paths = [
                fp.replace("piano_midi.mid", "metadata_tivo.json")
                for fp, _ in self.chunk_paths_and_idxs
            ]
            validate_paths(self.metadata_paths, expected_extension=".json")

    def chunker(self, score: Score) -> list[list[int]]:
        """Chunks a symusic.Score object into chunks of `max_seq_len` size"""
        # Preprocess the music file with the tokenizer
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
            # Apply our own preprocessing to the score
            preprocessed_score = preprocess_score(
                score,
                min_duration_ticks=self.min_duration_ticks,
                overlap_ticks=self.overlap_ticks
            )
            # Convert into chunks
            chunked_file = self.chunker(preprocessed_score)
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
        # Apply our own preprocessing to the score
        preprocessed_score = preprocess_score(
            score,
            min_duration_ticks=self.min_duration_ticks,
            overlap_ticks=self.overlap_ticks
        )
        # Convert the whole score into chunks
        chunked = self.chunker(preprocessed_score)
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
        # Add the conditioning tokens in to the start of the sequence, if required
        if self.do_conditioning:
            # Read the metadata JSON file (with a large cache to prevent redundant reads)
            metadata_read = utils.read_json_cached(self.metadata_paths[idx])
            # Grab the condition tokens for this track
            condition_tokens = get_conditions_for_track(self.condition_mapping, metadata_read, self.tokenizer)
            # If we actually have condition tokens
            if len(condition_tokens) > 0:
                # Add them to the sequence, preserving the target length and the BOS token (if this is present)
                input_ids, targets = add_condition_tokens_to_sequence(
                    sequence=input_ids,
                    condition_tokens=condition_tokens,
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

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"Exhaustive dataloader with {len(self.files_paths)} files, {len(self.chunk_paths_and_idxs)} chunks."


if __name__ == "__main__":
    from miditok import REMI

    # Get a tokenizer with default arguments
    token_factory = REMI()
    # Get filepaths for all MIDI files in the /data/raw/ directories
    midi_paths = utils.get_data_files_with_ext(ext="**/*.mid")[:100]

    # Create condition mapping
    cmap = {c: get_condition_special_tokens(c) for c in ["genres", "pianist"]}
    for mapping in cmap.values():
        for token in mapping.values():
            token_factory.add_to_vocab(token)

    # Test out our random chunking dataloader
    dm = DatasetMIDIRandomChunk(
        token_factory,
        midi_paths,
        max_seq_len=100,
        do_augmentation=True,
        do_conditioning=True,
        condition_mapping=cmap
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
        condition_mapping=cmap
    )
    print(dm)
    for i in range(len(dm)):
        item = dm.__getitem__(i)
        print(f'Item {i}, inputs shape {item["input_ids"].size(0)}, labels shape {item["labels"].size(0)}')
