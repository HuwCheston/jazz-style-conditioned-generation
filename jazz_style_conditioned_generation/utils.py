#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility classes, functions, and variables used across the entire pipeline"""

import json
import os
import random
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from time import time, sleep
from typing import ContextManager

import numpy as np
import torch
import transformers
from loguru import logger

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42

PIANO_KEYS = 88
FPS = 100
MIDI_OFFSET = 21
MAX_VELOCITY = 127
MIDDLE_C = 60
OCTAVE = 12

MAX_SEQUENCE_LENGTH = 1024
# This is important: it ensures that two chunks overlap slightly, to allow a causal chain between the
#  end of one chunk and the beginning of the next. The MIDITok default is 1: increasing this seems to work better
CHUNK_OVERLAP_BARS = 2


def seed_everything(seed: int = SEED) -> None:
    """Sets all random seeds for reproducible results."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # safe to call even if cuda is not available
    random.seed(seed)
    np.random.seed(seed)
    transformers.set_seed(seed)


def total_parameters(layer) -> int:
    return sum(p.numel() for p in layer.parameters())


@contextmanager
def timer(name: str) -> ContextManager[None]:
    """Print out how long it takes to execute the provided block."""
    from loguru import logger
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


def get_project_root() -> str:
    """Returns the root directory of the project"""
    return os.path.abspath(os.curdir)


@lru_cache(maxsize=None)
def get_data_files_with_ext(midi_dir_from_root: str = "data/raw", ext: str = "**/*.mid") -> list[str]:
    """Gets the filepaths recursively for all files with a given extension inside midi_dir_from_root"""
    return [str(p) for p in Path(os.path.join(get_project_root(), midi_dir_from_root)).glob(ext)]


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
