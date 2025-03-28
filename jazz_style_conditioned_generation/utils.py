#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility classes, functions, and variables used across the entire pipeline"""

import json
import os
import random
import string
from contextlib import contextmanager
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from time import time, sleep
from typing import ContextManager, Callable

import numpy as np
import torch
import transformers
from loguru import logger
from miditok import MusicTokenizer
from symusic import Score, Synthesizer, BuiltInSF3, dump_wav

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42

MIDI_PIANO_PROGRAM = 0
PIANO_KEYS = 88
MIDI_OFFSET = 21
MAX_VELOCITY = 127
MIDDLE_C = 60
OCTAVE = 12

# We'll resample input files to use these values
TICKS_PER_QUARTER = 500
TEMPO = 120
TIME_SIGNATURE = 2

MAX_SEQUENCE_LENGTH = 2048  # close to Music Transformer
# This is important: it ensures that two chunks overlap slightly, to allow a causal chain between the
#  end of one chunk and the beginning of the next. The MIDITok default is 1: increasing this seems to work better
CHUNK_OVERLAP_BARS = 8


def get_project_root() -> str:
    """Returns the root directory of the project"""
    # Possibly the root directory, but doesn't work when running from the CLI for some reason
    poss_path = str(Path(__file__).parent.parent)
    # The root directory should always have these files (this is pretty hacky)
    if all(fp in os.listdir(poss_path) for fp in ["config", "checkpoints", "data", "outputs", "setup.py"]):
        return poss_path
    else:
        return os.path.abspath(os.curdir)


# These are the names of all the datasets we're using: one "folder" per dataset
DATASETS = [i for i in os.listdir(os.path.join(get_project_root(), "data/raw")) if ".gitkeep" not in i]
DATASETS_WITH_TIVO = ["jtd", "pijama", "pianist8"]


def seed_everything(seed: int = SEED) -> None:
    """Sets all random seeds for reproducible results."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # safe to call even if cuda is not available
    random.seed(seed)
    np.random.seed(seed)
    transformers.set_seed(seed)


def total_parameters(layer: torch.nn.Module) -> int:
    """Gets total number of parameters for a pytorch module"""
    return sum(p.numel() for p in layer.parameters())


@contextmanager
def timer(name: str) -> ContextManager[None]:
    """Print out how long it takes to execute the provided block."""
    start = time()
    try:
        yield
    except Exception as e:
        end = time()
        logger.warning(f"Took {end - start:.2f} seconds to {name} and raised {e}.")
        raise e
    else:
        end = time()
        logger.debug(f"Took {end - start:.2f} seconds to {name}.")


@lru_cache(maxsize=None)
def get_data_files_with_ext(dir_from_root: str = "data/raw", ext: str = "**/*.mid") -> list[str]:
    """Gets the filepaths recursively for all files with a given extension inside midi_dir_from_root"""
    return [str(p) for p in Path(os.path.join(get_project_root(), dir_from_root)).glob(ext)]


@lru_cache(maxsize=None)
def read_json_cached(json_fpath: str) -> dict:
    """Reads metadata for a given track with a cache to prevent redundant operations"""
    assert os.path.isfile(json_fpath), f"Could not find JSON at {json_fpath}!"
    with open(json_fpath, 'r') as f:
        metadata = json.load(f)
    return metadata


def wait(secs: int):
    """Little decorator that adds a wait in to prevent too many API calls being made too quickly"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            ret = func(*args, **kwargs)
            sleep(secs)
            return ret

        return wrapper

    return decorator


def update_dictionary(d1: dict, d2: dict, overwrite: bool = False) -> dict:
    """Update missing key-value pairs in `d1` with key-value pairs in `d2`"""
    for k, v in d2.items():
        if not overwrite:
            if k not in d1.keys():
                d1[k] = v
        else:
            d1[k] = v
    return d1


def get_chunk_number_from_filepath(chunk_filepath: str):
    """Get the number of a MIDI chunk from a filepath: a value of 0 means this is the start of the whole track"""
    try:
        return int(chunk_filepath.split(os.path.sep)[-1].split("_")[-1].replace(".mid", ""))
    except (ValueError, IndexError):
        logger.warning(f"Couldn't get chunk number from filepath {chunk_filepath}, is is malformed?")
        return None


def add_to_tensor_at_idx(input_tensor: torch.tensor, insert_tensor: torch.tensor, insert_idx: int = 1) -> torch.tensor:
    """Adds in a tensor to an input at the given `insert_idx`"""
    return torch.cat([input_tensor[:insert_idx], insert_tensor, input_tensor[insert_idx:]])


def now() -> str:
    """Returns the current time, formatted nicely"""
    return datetime.now().strftime('%y_%m_%d_%H_%M_%S')


def random_probability() -> float:
    """Returns a random float between 0 and 1, useful in e.g. deciding whether to apply augmentation or not"""
    return random.uniform(0, 1)


def get_pitch_range(score: Score) -> tuple[int, int]:
    """Returns the pitch range of a Symusic Score object"""
    pitches = [i.pitch for i in score.tracks[0].notes]
    min_pitch, max_pitch = min(pitches), max(pitches)
    # Sanity check the pitch range: should be within the range of the piano
    assert min_pitch >= MIDI_OFFSET
    assert max_pitch <= (MIDI_OFFSET + PIANO_KEYS)
    return min_pitch, max_pitch


def remove_punctuation(s: str) -> str:
    """Removes punctuation from a given input string `s`"""
    return ''.join(
        c for c in str(s).translate(str.maketrans('', '', string.punctuation)).replace('’', '')
        if c in string.printable
    )


def write_json(metadata_dict: dict, filepath: str) -> None:
    """Dumps a dictionary as a JSON in provided location"""
    if not filepath.endswith(".json"):
        filepath += ".json"
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(metadata_dict, f, indent=4, ensure_ascii=False, sort_keys=False)


def string_to_bool(v) -> bool:
    """Coerces an argument to a boolean"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError(f'Could not coerce argument {v} to a boolean.')


def synthesize_score(score: Score, out_path: str = None) -> np.ndarray:
    """Synthesises a symusic.Score object and returns the waveform: if `out_path` is given, will also save the audio."""
    sf_path = BuiltInSF3.MuseScoreGeneral().path(download=True)
    synth = Synthesizer(sf_path=sf_path, sample_rate=44100)
    audio = synth.render(score, stereo=True)
    if out_path is not None:
        dump_wav(out_path, audio, sample_rate=44100, use_int16=True)
    return audio


def get_original_function(func: Callable) -> Callable:
    """Traverses a function signature to get the original function when wrapped with multiple decoratoes"""
    while hasattr(func, "__wrapped__"):
        func = func.__wrapped__
    return func


def base_round(x: float, base: int = 10) -> int:
    """Rounds a number to the nearest base"""
    return int(base * round(float(x) / base))


def weighted_sample(to_sample: list[str], probabilities: list[int], n_to_sample: int) -> list[int]:
    """Takes a weighted sample of N elements from to_sample. If N < |to_sample|, N = |to_sample|"""
    total = sum(probabilities)
    # We need to "softmax" our probabilities for NumPy (make them sum to one)
    if total != 1.:
        probabilities = [w / total for w in probabilities] if total != 0 else [0] * len(probabilities)
    # If we're trying to sample too many elements, reduce the number we're trying to sample
    if n_to_sample > len(to_sample):
        n_to_sample = len(to_sample)
    # Make the random sample
    sampled = np.random.choice(to_sample, n_to_sample, p=probabilities, replace=False)
    # Sort to maintain the same order as the original input
    return sorted(sampled, key=to_sample.index)


def validate_paths(filepaths: list[str], expected_extension: str = ".mid"):
    """Validates that all paths exist on disk and have an expected extension"""
    for file in filepaths:
        assert os.path.isfile(file), f"File {file} does not exist on the disk!"
        assert file.endswith(expected_extension), f"File {file} does not have expected extension {expected_extension}!"


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


def decode_bpe_sequence(sequence: torch.Tensor, tokenizer: MusicTokenizer) -> torch.Tensor:
    """Decodes a sequence of BPE-encoded token IDs into "raw" token IDs"""
    # Convert list of integers to a torch tensor
    if not isinstance(sequence, torch.Tensor):
        sequence = torch.tensor(sequence, dtype=torch.long)

    # Input was passed as (sequence_len)
    if len(sequence.size()) == 1:
        # Coerce it to (batch_size, sequence_len)
        sequence = sequence.unsqueeze(0)

    # These two private functions are called inside MusicTokenizer.decode
    converted = tokenizer._convert_sequence_to_tokseq(sequence.cpu())
    for seq in converted:
        tokenizer._preprocess_tokseq_before_decoding(seq)
        # Sanity check that the de-tokenized ID is in the base vocab of our model
        for detok in seq:
            assert detok in list(tokenizer.vocab.values())
    # We need to pad the input to match the longest sequence length in the batch, as this may be ragged
    all_ids = [i.ids for i in converted]
    pad_size = max(len(i) for i in all_ids)
    # This applies right padding, so that the tensor becomes (batch_size, longest_detokenized_sequence)
    padded = [pad_sequence(s, pad_size, tokenizer.pad_token_id) for s in all_ids]
    return torch.tensor(padded)


if __name__ == "__main__":
    logger.info(f"Root directory: {get_project_root()}")
