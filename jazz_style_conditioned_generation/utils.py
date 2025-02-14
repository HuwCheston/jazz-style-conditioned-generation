#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility classes, functions, and variables used across the entire pipeline"""

import random
from contextlib import contextmanager
from pathlib import Path
from time import time
from typing import ContextManager

import numpy as np
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42

PIANO_KEYS = 88
FPS = 100
MIDI_OFFSET = 21
MAX_VELOCITY = 127
MIDDLE_C = 60
OCTAVE = 12


def seed_everything(seed: int = SEED) -> None:
    """Sets all random seeds for reproducible results."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


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


def get_project_root() -> Path:
    """Returns the root directory of the project"""
    return Path(__file__).absolute().parent.parent
