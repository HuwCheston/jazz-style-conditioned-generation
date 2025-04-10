#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Objective metrics for use during evaluation of generated or real examples"""

import os
import tempfile
from functools import wraps
from inspect import getfullargspec
from pathlib import Path
from typing import Callable, Any

import muspy
import numpy as np
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from loguru import logger
from miditok import MusicTokenizer
from symusic import Score
from tqdm import tqdm

from jazz_style_conditioned_generation import utils
from jazz_style_conditioned_generation.data.dataloader import DatasetMIDIConditionedNoOverlapChunks
from jazz_style_conditioned_generation.data.tokenizer import load_tokenizer, train_tokenizer


def accuracy_score(logits: torch.Tensor, labels: torch.Tensor, tokenizer: MusicTokenizer) -> torch.Tensor:
    """Calculate accuracy between predicted + target labels while handling BPE decoding"""
    assert hasattr(tokenizer, "bpe_token_mapping"), "Must have set the `bpe_token_mapping` attribute!"
    # Tracks the length of all sequences after BPE decoding + the number of hits
    total_seq_len, hits = 0, 0
    # Iterate over each item in the batch
    for logits_item, labels_item in zip(logits, labels):
        # Convert the raw logits into softmaxed predictions
        predictions_item = torch.argmax(torch.softmax(logits_item, dim=-1), dim=-1)
        this_seq_len = 0  # tracks the length of the current prediction
        # Iterate over maybe-BPE encoded predicted + target tokens
        for predict, targ in zip(predictions_item, labels_item):
            # Convert BPE encoded tokens into a list of non-BPE encoded tokens
            predict_decode = tokenizer.bpe_token_mapping[predict.item()]
            targ_decode = tokenizer.bpe_token_mapping[targ.item()]
            # The length of both lists might be different
            #  If we have more target labels than predicted labels
            if len(predict_decode) < len(targ_decode):
                # Pad the sequence by continuing to predict the final token
                overlap = len(targ_decode) - len(predict_decode)
                predict_decode = predict_decode + [predict_decode[-1] for _ in range(overlap)]
            #  If we have fewer target labels than predicted labels
            elif len(predict_decode) > len(targ_decode):
                # Truncate the predicted labels to match the length of the predicted labels
                predict_decode = predict_decode[:len(targ_decode)]
            # Sanity check that everything should now be identical
            assert len(predict_decode) == len(targ_decode)
            # Iterate over INDIVIDUAL, non-BPE predicted + target labels
            for p, t in zip(predict_decode, targ_decode):
                # Skip over padded label tokens
                if t == tokenizer.pad_token_id:
                    continue
                else:
                    # Append to our counters
                    this_seq_len += 1
                    total_seq_len += 1
                    # If the predicted label is equal to the target label
                    if p == t:
                        hits += 1
                # Break out of the loop once we've reached the target length after BPE decoding
                if this_seq_len >= utils.MAX_SEQUENCE_LENGTH:
                    break
    # Simple accuracy measurement, return as tensor for compatibility with e.g. loss metric
    return torch.tensor(hits / total_seq_len)


def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor, tokenizer: MusicTokenizer) -> torch.Tensor:
    """Just implements the vanilla cross entropy loss from torch with some reshaping"""
    return F.cross_entropy(
        # Reshapes logits to (batch_size * sequence_len, vocab_size)
        logits.reshape(logits.shape[0] * logits.shape[1], -1).to(torch.float),
        # Reshapes targets to (batch_size * sequence_len)
        labels.flatten().to(torch.long),
        ignore_index=tokenizer.pad_token_id
    )


def _symusic_to_muspy(score: Score) -> muspy.Music:
    """Converts a symusic.Score object to a muspy.Music object by dumping a temporary MIDI file and cleaning up after"""
    # Dump the score object to a temporary midi file
    tempfname = os.path.join(utils.get_project_root(), next(tempfile._get_candidate_names()) + ".mid")
    score.dump_midi(tempfname)
    # Load in as a muspy object with the same resolution
    mp = muspy.read_midi(tempfname)
    # Sanity check everything
    assert mp.resolution == score.ticks_per_quarter  # Should have the same resolution
    assert len(mp.tracks[0].notes) == len(score.tracks[0].notes)  # Should have the same number of notes
    # TODO: can be flaky here, should probably be assert math.isclose
    # assert mp.get_end_time() == score.end()  # Should have the same end time
    # Tidy up by removing the temporary midi file
    os.remove(tempfname)
    return mp


def coerce_to_muspy(func: Callable):
    """Coerces input types to a MusPy object, to be used as a decorator"""

    @wraps(func)
    def _coerce_type(score: Any):
        if isinstance(score, Score):
            return func(_symusic_to_muspy(score))
        elif isinstance(score, str):
            return func(muspy.read_midi(score))
        elif isinstance(score, muspy.Music):
            return func(score)
        else:
            raise TypeError(f"Expected `score` to be `symusic.Score`, `muspy.Music`, or `str`, but got {type(score)}")

    return _coerce_type


def catch_zero_division(func: Callable):
    """Little decorator that returns NaN for any function that raises a ZeroDivisionError"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ZeroDivisionError:
            return np.nan

    return wrapper


@coerce_to_muspy
def event_density(score: Any) -> int:
    """Returns the density of the input in terms of number of notes"""
    try:
        return len(score.tracks[0].notes)
    except IndexError:
        return 0


@coerce_to_muspy
def pitch_class_entropy(score: Any) -> float:
    return muspy.pitch_class_entropy(score)


@coerce_to_muspy
def pitch_range(score: Any) -> int:
    return muspy.pitch_range(score)


@coerce_to_muspy
def polyphony(score: Any) -> int:
    return muspy.polyphony(score)


@coerce_to_muspy
def number_of_pitches(score: Any) -> int:
    return muspy.n_pitches_used(score)


@coerce_to_muspy
def number_of_pitch_classes(score: Any) -> int:
    return muspy.n_pitch_classes_used(score)


@catch_zero_division
def _tone_spans(sorted_pitches: list[int], threshold: int = utils.OCTAVE * 2):
    above_threshold = 0
    for p1, p2 in zip(sorted_pitches, sorted_pitches[1:]):
        if abs(p2 - p1) > threshold:
            above_threshold += 1
    return above_threshold / (len(sorted_pitches) - 1)


@coerce_to_muspy
def tone_spans(score: Any) -> float:
    """Returns the ratio of adjacent notes that have intervals greater than a given threshold"""
    sorted_notes = sorted(score.tracks[0].notes, key=lambda x: x.time)
    return _tone_spans([i.pitch for i in sorted_notes])


@catch_zero_division
def _consecutive_repetitions(sorted_pitches: list[int]) -> float:
    """Given a list of integers corresponding to pitch or p-class, calculate the number of consecutive repetitions"""
    pitch_repetitions = 0
    for p1, p2 in zip(sorted_pitches, sorted_pitches[1:]):
        if p2 == p1:
            pitch_repetitions += 1
    return pitch_repetitions / (len(sorted_pitches) - 1)


@coerce_to_muspy
def consecutive_pitch_repetitions(score: Any) -> float:
    """Returns the ratio of adjacent notes that are repetitions of the same pitch"""
    sorted_notes = sorted(score.tracks[0].notes, key=lambda x: x.time)
    return _consecutive_repetitions([i.pitch for i in sorted_notes])


@coerce_to_muspy
def consecutive_pitch_class_repetitions(score: Any) -> float:
    """Returns the ratio of adjacent notes that are repetitions of the same pitch class"""
    sorted_notes = sorted(score.tracks[0].notes, key=lambda x: x.time)
    as_pitch_classes = [i.pitch % utils.OCTAVE for i in sorted_notes]
    return _consecutive_repetitions(as_pitch_classes)


@catch_zero_division
def rhythmic_variation(token_ids: Any, tokenizer: MusicTokenizer) -> float:
    """Return the ratio of unique time or duration tokens for a sequence with respect to a tokenizer"""
    # TODO: see if this is possible to implement?
    if tokenizer.is_trained:
        return np.nan

    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    if len(token_ids) == 0:
        return 0.
    # Need to remove the batch dimension here
    if isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    tokenizer_time_values = [i for i in tokenizer.vocab.keys() if "time" in i.lower() or "duration" in i.lower()]
    tokenized_ids = [tokenizer[i] for i in token_ids]
    time_tokens = [t for t in tokenized_ids if "time" in t.lower() or "duration" in t.lower()]
    return len(set(time_tokens)) / len(set(tokenizer_time_values))


ALL_METRICS = [
    event_density,
    pitch_class_entropy,
    pitch_range,
    polyphony,
    number_of_pitches,
    number_of_pitch_classes,
    tone_spans,
    consecutive_pitch_repetitions,
    consecutive_pitch_class_repetitions,
    rhythmic_variation
]
ALL_METRIC_NAMES = [
    "event_density",
    "pitch_class_entropy",
    "pitch_range",
    "polyphony",
    "number_of_pitches",
    "number_of_pitch_classes",
    "tone_spans",
    "consecutive_pitch_repetitions",
    "consecutive_pitch_class_repetitions",
    "rhythmic_variation"
]


def return_empty_on_error():
    res_for_track = {}
    for func in ALL_METRIC_NAMES:
        res_for_track[func] = np.nan
    return res_for_track


def compute_metrics_for_sequence(token_ids: torch.Tensor, tokenizer: MusicTokenizer):
    """Given a tensor of token ids from a sequence, compute all evaluation metrics"""
    # Need to add the batch dimension in if this isn't present already
    if len(tuple(token_ids.size())) == 1:
        token_ids = token_ids.unsqueeze(0)
    # We'll store results for this track in here
    res_for_track = {}
    # Decode the IDs into a symusic.Score object, then convert this to Muspy
    try:
        decoded = tokenizer.decode(token_ids)
    # TODO: this seems to be an error with PerTok
    except ValueError as e:
        logger.warning(f"Got error when decoding tokens! All metrics will be missing! {e}")
        return return_empty_on_error()
    # If we decode the tokens into an empty sequence
    if len(decoded.tracks) == 0:
        logger.warning(f"Sequence decoded into an empty score. All metrics will be missing! Sequence {token_ids}")
        # We need to return None for all metrics, so that this sequence will be skipped if we go on to aggregate
        return return_empty_on_error()
    # Otherwise, we can safely coerce the score to muspy
    coerced_to_muspy = _symusic_to_muspy(decoded)
    # Iterate over all the metrics we want to calculate
    for func_to_call, func in zip(ALL_METRICS, ALL_METRIC_NAMES):
        func_to_call = utils.get_original_function(func_to_call)
        needed_args = getfullargspec(func_to_call).args
        # Function is expecting a muspy.Music object
        if "score" in needed_args:
            func_res = func_to_call(coerced_to_muspy)
        # Function is expecting the raw tokens and the tokenizer
        elif "token_ids" in needed_args:
            func_res = func_to_call(token_ids, tokenizer)
        # Function has an incorrect signature
        else:
            raise AttributeError(f"Unknown function signature {needed_args} for function name {func}")
        res_for_track[func] = func_res
    return res_for_track


def compute_metrics_for_sequences(token_ids_list: list[torch.Tensor], tokenizer: MusicTokenizer) -> dict:
    """Computes evaluation metrics for multiple sequences in parallel and aggregates into a single dictionary"""
    # Get values for all metrics from every token id in the list
    with Parallel(n_jobs=-1) as par:
        all_res = par(
            delayed(compute_metrics_for_sequence)(token_ids, tokenizer)
            for token_ids in tqdm(token_ids_list, desc="Getting evaluation metric results...")
        )
    # Aggregate across values obtained from every chunk
    return aggregate_evaluation_metrics(all_res)


def aggregate_evaluation_metrics(evaluation_results: list[dict]) -> dict:
    """Aggregates evaluation results obtained from multiple sequences into a single dictionary"""
    all_aggs = {}
    for k in evaluation_results[0]:
        k_vals = [i[k] for i in evaluation_results]
        for agg_name, agg_func in zip(["mean", "std"], [np.nanmean, np.nanstd]):
            agg_res = agg_func(k_vals)
            all_aggs[f"{k}_{agg_name}"] = agg_res
    return all_aggs


def compute_metrics_for_dataset(dataset_tracks: list[str], tokenizer: MusicTokenizer, **dataset_kwargs) -> dict:
    """Given a list of tracks in a dataset and a tokenizer, compute all evaluation metrics and aggregate"""
    if isinstance(dataset_tracks[0], Path):
        dataset_tracks = [str(dt) for dt in dataset_tracks]
    # Create the dataset object, which will chunk every track into non-overlapping chunks of N tokens
    dataset = DatasetMIDIConditionedNoOverlapChunks(
        tokenizer,
        dataset_tracks,
        max_seq_len=utils.MAX_SEQUENCE_LENGTH,
        do_augmentation=False,
        do_conditioning=False,
        **dataset_kwargs
    )
    # Get values for all metrics from every chunk in the dataset in parallel
    with Parallel(n_jobs=-1) as par:
        all_res = par(
            delayed(compute_metrics_for_sequence)(td["input_ids"], tokenizer)
            for td in tqdm(dataset, desc="Getting evaluation metric results...")
        )
    # Aggregate across values obtained from every chunk
    return aggregate_evaluation_metrics(all_res)


if __name__ == "__main__":
    token = load_tokenizer()
    jtd = utils.get_data_files_with_ext("data/raw/jtd", "**/*.mid")[:10]
    train_tokenizer(token, jtd)
    cm = compute_metrics_for_dataset(jtd, token)

    print('Metrics for 10 JTD tracks: ')
    print(cm)
