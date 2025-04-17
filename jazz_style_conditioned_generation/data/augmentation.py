#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Data augmentation for symusic objects"""

import numpy as np
from miditok.data_augmentation import augment_score
from symusic import Score
from symusic.core import Second

from jazz_style_conditioned_generation import utils

PITCH_AUGMENT_RANGE = range(-3, 4)  # as in Music Transformer
DURATION_AUGMENT_RANGE = [0.95, 0.975, 1.0, 1.025, 1.05]  # as in Music Transformer
VELOCITY_AUGMENT_RANGE = [0]  # for back compatibility, before we added in velocity augmentation


def get_pitch_augmentation_value(score: Score, pitch_augmentation_range: list) -> int:
    """Gets a pitch augmentation value that can be applied to a Score without exceeding the limits of the keyboard"""
    # Get the minimum and maximum pitch from the score
    min_pitch, max_pitch = utils.get_pitch_range(score)
    # Default values
    pitch_augment, min_pitch_augmented, max_pitch_augmented = 0, 0, 1000
    # Keep iterating until we have an acceptable pitch augmentation value
    while min_pitch_augmented < utils.MIDI_OFFSET or max_pitch_augmented > (utils.MIDI_OFFSET + utils.PIANO_KEYS) - 1:
        # Get a possible pitch augment value
        pitch_augment = np.random.choice(pitch_augmentation_range, 1)[0]
        # Add this to the min and max pitch
        min_pitch_augmented = min_pitch + pitch_augment
        max_pitch_augmented = max_pitch + pitch_augment
    return pitch_augment


def _data_augmentation_deterministic(
        score: Score,
        pitch_augment_value: int,
        duration_augment_value: float,
        velocity_augment_value: int
) -> Score:
    """Applies pitch and duration augmentation with specified values to a Score object"""
    # We can just apply the pitch augmentation directly using the MIDITok function
    augmented = augment_score(
        score,
        pitch_offset=pitch_augment_value,
        augment_copy=True,
        duration_offset=0.,
        velocity_offset=velocity_augment_value,
        velocity_range=(1, utils.MAX_VELOCITY)
    )
    # Sanity check: pitches should be within the range of the piano keyboard
    aug_min, aug_max = utils.get_pitch_range(augmented)
    assert aug_min >= utils.MIDI_OFFSET
    assert aug_max <= utils.MIDI_OFFSET + utils.PIANO_KEYS
    # We need to use the symusic pretty_midi-like function to do duration augmentation
    new_end = augmented.end() * duration_augment_value
    # With timing in ticks (integers), we need to round the new end time to the nearest integer
    # Otherwise, with timing in seconds (float), we just use the raw value
    if not isinstance(augmented.ttype, Second):
        new_end = round(new_end)
    return augmented.adjust_time(
        [augmented.start(), augmented.end()],
        [augmented.start(), new_end],
        inplace=False
    )


def data_augmentation(
        score: Score,
        pitch_augmentation_range: list = None,
        duration_augmentation_range: list = None,
        velocity_augmentation_range: list = None
) -> tuple[Score, float]:
    """Applies pitch and duration augmentation to a Score object"""
    if pitch_augmentation_range is None:
        pitch_augmentation_range = PITCH_AUGMENT_RANGE
    if duration_augmentation_range is None:
        duration_augmentation_range = DURATION_AUGMENT_RANGE
    if velocity_augmentation_range is None:
        velocity_augmentation_range = VELOCITY_AUGMENT_RANGE

    # Get random augmentation value from the ranges provided
    pitch_augment = get_pitch_augmentation_value(score, pitch_augmentation_range)
    duration_augment = np.random.choice(duration_augmentation_range)
    velocity_augment = np.random.choice(velocity_augmentation_range)
    # Apply data augmentation with the randomly selected values
    #  Also need to return the amount we're augmenting by, so we can adjust our tempo
    return _data_augmentation_deterministic(score, pitch_augment, duration_augment, velocity_augment), duration_augment
