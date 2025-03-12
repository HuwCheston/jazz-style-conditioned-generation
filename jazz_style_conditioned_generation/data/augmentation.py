#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Data augmentation for symusic objects"""

import numpy as np
from miditok.data_augmentation import augment_score
from symusic import Score

import utils

PITCH_AUGMENT_RANGE = range(-3, 4)  # as in Music Transformer
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
