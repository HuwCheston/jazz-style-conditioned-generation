#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Trains a Tokenizer in MIDITok and dumps a .json with all parameters"""

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Sequence, Union

import numpy as np
from loguru import logger
from miditok import MusicTokenizer, TokSequence, TokenizerConfig
from miditok.attribute_controls import create_random_ac_indexes
from miditok.constants import (
    SCORE_LOADING_EXCEPTION,
    CURRENT_MIDITOK_VERSION,
    CURRENT_SYMUSIC_VERSION,
    CURRENT_TOKENIZERS_VERSION
)
from miditok.tokenizations import REMI, MIDILike, TSD, Structured, PerTok
from miditok.tokenizer_training_iterator import TokTrainingIterator
from miditok.utils import convert_ids_tensors_to_list
from miditok.utils.utils import np_get_closest
from symusic import Score, Tempo, TimeSignature, Note, Track
from tqdm import tqdm

from jazz_style_conditioned_generation import utils
from jazz_style_conditioned_generation.data.conditions import INCLUDE
from jazz_style_conditioned_generation.data.scores import load_score, preprocess_score

DEFAULT_TOKENIZER_CONFIG = {
    "pitch_range": (utils.MIDI_OFFSET, utils.MIDI_OFFSET + utils.PIANO_KEYS),
    "beat_res": {(0, utils.TIME_SIGNATURE): 100 // utils.TIME_SIGNATURE},  # 100 tokens per "bar", 10ms each
    "num_velocities": 32,
    "special_tokens": [
        "PAD",  # add for short inputs to ensure consistent sequence length for all inputs
        "BOS",  # beginning of sequence
        "EOS",  # end of sequence
        # "MASK",  # prevent attention to future tokens
    ],
    "use_chords": False,
    "use_rests": False,
    "use_tempos": False,
    "use_time_signatures": False,
    "use_programs": False,
    "use_sustain_pedals": False,
    "use_pitch_bends": False,
    "use_velocities": True,
    # "chord_maps": constants.CHORD_MAPS,  # TODO: think more about this
    "remove_duplicated_notes": True,
    "encode_ids_split": "no",
    "use_pitchdrum_tokens": False,
    "programs": [0],  # only piano
}
DEFAULT_VOCAB_SIZE = 1000
DEFAULT_TRAINING_METHOD = "BPE"
DEFAULT_TOKENIZER_CLASS = "tsd"

OUTPUT_MIDI_DIR = os.path.join(utils.get_project_root(), 'outputs/midi/tokenized')


class CustomTokenizerConfig(TokenizerConfig):
    def __init__(self, *args, **kwargs):
        self.time_range = kwargs.pop("time_range", (0.01, 2.5))
        self.time_factor = kwargs.pop("time_factor", 1.03)
        super().__init__(*args, **kwargs)


class CustomTSD:
    """
    Custom implementation of Time-Shift Duration tokenizations, with:

    1) Performance timing (using symusic.Score` objects with `ttype=Second`, and that may not contain bar/tempo info)
    2) Multiple duration tokens for notes longer than `max(Duration_Token)`.

    Everything else should be pretty much the same as the MIDITok API. Use the CustomTokenizerConfig class to pass in
    the tokenizer parameters. Set `time_range` and `time_factor` to specify the minimum and maximum time, and the
    factor used to between successive times, where `time_{i} = time_{i-1} * factor`. Set `time_factor = 1` for linear
    distance between successive times.

    NB. There are lots of hard-coded assumptions currently, such that the MIDI files will only contain one piano track
    and that they will not contain embedded tempo or time signature information. These will be ignored if provided.

    """

    def __init__(
            self,
            tokenizer_config: Union[TokenizerConfig, CustomTokenizerConfig] = None,
            params: str | Path | None = None
    ):
        # vocab of prime tokens, can be viewed as unique chars / bytes
        self._vocab_base = {}
        # the other way, to decode id (int) -> token (str)
        self.__vocab_base_inv = {}

        # No training for this tokenizer
        self._model = None

        # Initialize config
        # Loading params, or initialising them from args
        if params is not None:
            self._load_from_json(params)
        elif tokenizer_config is None:
            self.config = CustomTokenizerConfig()
        else:
            self.config = tokenizer_config

        if len(self.vocab) == 0:
            self.__create_vocabulary()

    def _load_from_json(self, file_path: str | Path):
        """Load the parameters of the tokenizer from a config file"""
        params = utils.read_json_cached(file_path)

        self.config = CustomTokenizerConfig()
        config_attributes = list(self.config.to_dict().keys())
        old_add_tokens_attr = {
            "Chord": "use_chords",
            "Rest": "use_rests",
            "Tempo": "use_tempos",
            "TimeSignature": "use_time_signatures",
            "Program": "use_program",
        }

        # Overwrite config attributes
        for key, value in params.items():
            if key in ["tokenization", "miditok_version", "_model", "_vocab_base_byte_to_token", "has_bpe"]:
                continue
            if key == "_vocab_base":
                self._vocab_base = value
                self.__vocab_base_inv = {v: k for k, v in value.items()}
                continue
            if key == "config":
                if "chord_maps" in value:
                    value["chord_maps"] = {
                        chord_quality: tuple(chord_map)
                        for chord_quality, chord_map in value["chord_maps"].items()
                    }
                for beat_res_key in ["beat_res", "beat_res_rest"]:
                    # check here for previous versions (< v2.1.5)
                    if beat_res_key in value:
                        try:
                            value[beat_res_key] = {
                                tuple(map(int, beat_range.split("_"))): res
                                for beat_range, res in value[beat_res_key].items()
                            }
                        except AttributeError:
                            continue
                value["time_signature_range"] = {
                    int(res): beat_range
                    for res, beat_range in value["time_signature_range"].items()
                }
                # Rest param < v2.1.5
                if "rest_range" in value:
                    value["rest_range"] = {
                        (0, value["rest_range"][1]): value["rest_range"][0]
                    }
                if "additional_params" not in value:
                    value["additional_params"] = {}
                value = CustomTokenizerConfig.from_dict(value)
            if key in config_attributes:
                if key == "beat_res":
                    value = {
                        tuple(map(int, beat_range.split("_"))): res
                        for beat_range, res in value.items()
                    }
                elif key == "time_signature_range":
                    value = {int(res): beat_range for res, beat_range in value.items()}
                # Convert old attribute from < v2.1.0 to new for TokenizerConfig
                elif key in old_add_tokens_attr:
                    key = old_add_tokens_attr[key]
                setattr(self.config, key, value)
                continue
            setattr(self, key, value)

    def to_dict(self) -> dict:
        """Return the serializable dictionary form of the tokenizer."""
        # Don't need to account for trained tokenizer here
        return {
            "config": self.config.to_dict(serialize=True),
            "tokenization": self.__class__.__name__,
            "miditok_version": CURRENT_MIDITOK_VERSION,
            "symusic_version": CURRENT_SYMUSIC_VERSION,
            "hf_tokenizers_version": CURRENT_TOKENIZERS_VERSION,
        }

    def save(
            self,
            out_path: str | Path,
            additional_attributes: dict | None = None,
            filename: str | None = "tokenizer.json",
    ) -> None:
        """Save tokenizer in a JSON file, useful to keep track of how a dataset has been tokenized."""
        tokenizer_dict = self.to_dict()

        if additional_attributes:
            tokenizer_dict.update(additional_attributes)

        out_path = Path(out_path)
        if out_path.is_dir() or "." not in out_path.name:
            out_path /= filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as outfile:
            json.dump(tokenizer_dict, outfile, indent=4)

    def _create_base_vocabulary(self) -> list[str]:
        """Create the vocabulary, as a list of string tokens. Each token is given as the form `"Type_Value"`"""
        vocab = []
        # Pitch
        vocab += [f"Pitch_{f}" for f in range(*self.config.pitch_range)]
        # Velocity
        vocab += [f"Velocity_{f}" for f in self.velocities]
        # Duration
        vocab += [f"Duration_{f}" for f in self.times]
        # TimeShift
        vocab += [f"TimeShift_{f}" for f in self.times]
        return vocab

    def __create_vocabulary(self):
        """Create the vocabulary of the tokenizer as a dictionary"""
        vocab = self._create_base_vocabulary()
        # MIDITok adds special tokens to vocabulary first, then normal tokens
        for tok in self.special_tokens:
            self.add_to_vocab(tok, special_token=True)
        for tok in vocab:
            self.add_to_vocab(tok)

    @property
    def vocab_model(self) -> None:
        """Return the vocabulary learned with BPE: is always None, as no training implemented yet"""
        return None

    @property
    def _model_name(self):
        """No training allowed for this tokenizer"""
        return "None"

    @property
    def special_tokens(self) -> list[str]:
        """Return the special tokens in the vocabulary"""
        return self.config.special_tokens

    @property
    def special_tokens_ids(self) -> list[int]:
        """Return the ids of the special tokens in the vocabulary"""
        return [self[token] for token in self.special_tokens]

    @property
    def io_format(self) -> tuple[str, ...]:
        """Return the I/O format of the tokenizer"""
        format_ = ["I", "T"]
        return tuple(d for d in format_)

    @property
    def velocities(self) -> np.ndarray:
        """Array of `num_velocity` MIDI velocities, from 0 to MAX_VELOCITY (127 in MIDI)"""
        # [1:] ensures that there is no velocity 0
        return np.linspace(0, utils.MAX_VELOCITY, self.config.num_velocities + 1, dtype=np.intc)[1:]

    @property
    def times(self) -> np.ndarray:
        """Array of time values, between min and max time and scaled by factor"""
        min_time, max_time = self.config.time_range
        # Sanity check from config
        assert min_time > 0. and max_time > 0., "`min_time` and `max_time` must be greater than 0"
        assert min_time < max_time, "`min_time` must be smaller than `max_time`"
        assert self.config.time_factor >= 1., "`time_factor` must be greater or equal to 1"
        # Linear spacing: can just use np.linspace
        if self.config.time_factor == 1.:
            times = np.linspace(min_time, max_time, int(max_time / min_time), dtype=np.float32)
        # Otherwise, using non-linear spacing
        else:
            times: list[float] = [min_time]
            while times[-1] < max_time:
                next_val = times[-1] * self.config.time_factor
                # Ensure a minimum step of `min_time`
                if next_val - times[-1] < min_time:
                    next_val = times[-1] + min_time
                # Break out once we've hit the maximum time
                if next_val >= max_time:
                    break
                # Round only when necessary for tokenization output (not internal calculations)
                times.append(next_val)
            times = sorted(times)
            # Add in the maximum time if the sequence doesn't end with it
            if times[-1] != max_time:
                times.append(max_time)
            times = np.array(times, dtype=np.float32)
        return np.rint(times * 1000).astype(int)

    @property
    def _times_tuple(self) -> tuple[int]:
        """Returns times as a tuple: useful for caching, where numpy arrays aren't supported"""
        return tuple(self.times)

    def add_to_vocab(self, token: str, special_token: bool = False, *_, **__) -> None:
        """Add an event to the vocabulary. Its ID will be the length of the vocab"""
        token_str = token if isinstance(token, str) else str(token)
        # Handling special tokens
        if special_token:
            parts = token_str.split("_")
            if len(parts) == 1:
                parts.append("None")
            elif len(parts) > 2:
                parts = ["-".join(parts[:-1]), parts[-1]]
            token = "_".join(parts)
        # Handling edge cases where a token is already in the vocabulary
        if token_str in self.vocab:
            token_id = self.vocab[token_str]
            logger.warning(f"Token {token} is already in the vocabulary at idx {token_id}")
        # Otherwise, add the token into the vocabulary
        else:
            id_ = len(self.vocab)
            self._vocab_base[token_str] = id_
            self.__vocab_base_inv[len(self.__vocab_base_inv)] = token_str

    def token_id_type(self, id_: int, ) -> str:
        """Return the type of the given token id."""
        token = self.__get_from_voc(id_)
        return token.split("_")[0]

    def _validate_tokens(self, tokens: list[str]):
        """Check that all tokens in a sequence are in the vocabulary"""
        for t in tokens:
            # Handle both ID (int) or token (str) input
            if isinstance(t, str):
                assert t in self.vocab.keys(), f"Token {t} is not in the vocabulary!"
            elif isinstance(t, int):
                assert t in self.__vocab_base_inv.keys(), f"Token {t} is not in the vocabulary!"
            # Unknown token type, need to raise an error
            else:
                raise TypeError(f"Expected either `str` or `int` type for token {t}, but got {type(t)}")

    @staticmethod
    def get_closest(array: np.ndarray, val: int) -> int:
        return array[np.argmin(np.abs(array - val))]

    @property
    def min_time(self):
        return min(self.times) / 2

    @staticmethod
    @lru_cache(maxsize=None)
    def decompose_time(value: int, times_tuple: tuple[int], min_time: int):
        """Keep decomposing a time value into its closest tokens, until `min_time` is reached"""
        times = np.fromiter(times_tuple, dtype=np.intc)
        result = []
        remaining = value

        while remaining > min_time:
            idx = np.searchsorted(times, remaining)
            if idx == 0:
                token = times[0]
            elif idx == len(times):
                token = times[-1]
            else:
                before, after = times[idx - 1], times[idx]
                token = before if abs(remaining - before) <= abs(remaining - after) else after

            result.append(token)
            remaining -= token

        return result

    def _score_to_tokens(self, score: Score) -> TokSequence:
        """Convert a **preprocessed** `symusic.Score` object to a sequence of tokens"""

        note_list = score.tracks[0].notes  # assuming only one track
        # Batch processing of pitches
        pitches = np.array([ev.pitch for ev in note_list])

        # Batch processing of start + duration times (in milliseconds!)
        starts_ms = np.round(np.array([ev.start for ev in note_list]) * 1000).astype(int)
        durations_ms = np.round(np.array([ev.duration for ev in note_list]) * 1000).astype(int)

        # Batch processing of velocities
        velocities = np.array([ev.velocity for ev in note_list])
        velocities_closest = np_get_closest(self.velocities, velocities)

        tokens = []  # store tokens
        previous_time = 0.  # cursor

        # Iterate over all notes
        for note_start, note_pitch, note_velocity, note_duration in zip(
                starts_ms, pitches, velocities_closest, durations_ms
        ):
            # Timeshift tokens (incremental update)
            for shift in self.decompose_time(note_start - previous_time, self._times_tuple, self.min_time):
                tokens.append(f"TimeShift_{shift}")
                previous_time += shift

            # Pitch token
            tokens.append(f"Pitch_{note_pitch}")

            # Velocity token: use the miditok function to get the closets velocity
            tokens.append(f"Velocity_{note_velocity}")

            # Duration
            for dt in self.decompose_time(note_duration, self._times_tuple, self.min_time):
                tokens.append(f"Duration_{dt}")

        # Tokenize as list of integers
        ids = self._ids_to_tokens(tokens)
        # Convert to MIDITok's format (for compatibility with MIDITok more broadly
        return [TokSequence(tokens=tokens, ids=ids, are_ids_encoded=False)]

    def encode(self, score, no_preprocess_score: bool = False, *_, **__) -> list[TokSequence]:
        """Tokenize a music file given as a `symusic.Score` or file path"""
        # Load the score
        if not isinstance(score, Score):
            score = load_score(score, as_seconds=True)
            # Preprocess it if required
            if not no_preprocess_score:
                score = preprocess_score(score)
        # Otherwise, we assume that the input score has already been preprocessed
        # Tokenize the score as a list of strings
        tokseq = self._score_to_tokens(score)
        for seq in tokseq:
            self.complete_sequence(seq)
        return tokseq

    def complete_sequence(self, seq: TokSequence) -> None:
        """Complete (inplace) a `miditok.TokSequence` by initialising empty attributes"""
        if len(seq.tokens) == 0:
            if len(seq.ids) > 0:
                seq.tokens = self._ids_to_tokens(seq.ids)
        if len(seq.ids) == 0:
            seq.ids = self._tokens_to_ids(seq.tokens)

    def _tokens_to_ids(self, tokens: list[str]) -> list[int]:
        """Convert a list of tokens (str) into their ids format (int)"""
        if len(tokens) == 0:
            return []
        elif isinstance(tokens[0], str):
            return [self.vocab[token] for token in tokens]
        else:
            # Can't handle multiple vocabularies yet
            raise NotImplementedError

    def _ids_to_tokens(self, ids: list[int], *_, **__) -> list[str]:
        """Convert a sequence of ids (int) to their tokens format (str)"""
        return [self[id_] for id_ in ids]

    def _tokens_to_score(self, tokens: list[str]) -> Score:
        self._validate_tokens(tokens)  # check all tokens in vocab
        # Create the score object
        score = Score(ttype="Second").resample(utils.TICKS_PER_QUARTER).to("Second")
        score.tracks = [Track(program=utils.MIDI_PIANO_PROGRAM, is_drum=False, ttype="Second")]
        # Iterate through all the passed tokens
        current_time = 0
        for ti, token in enumerate(tokens):
            tok_type, tok_val = token.split("_")
            # TimeShift tokens: add to the cursor
            if tok_type == "TimeShift":
                current_time += int(tok_val)
            # Pitch tokens
            elif tok_type == "Pitch":
                pitch = int(tok_val)
                try:
                    # Grab the velocity token: should always be next token after a pitch token
                    vel_type, vel = tokens[ti + 1].split("_")
                    velocity = int(vel)
                    # Grab the duration token(s): should always follow on from a velocity token
                    duration = 0
                    for maybe_duration_token in tokens[ti + 2:]:
                        dur_type, dur = maybe_duration_token.split("_")
                        # Continue gathering the duration up until we hit another type of token
                        if dur_type == "Duration":
                            duration += int(dur)
                        else:
                            break
                    # As long as we have velocities and durations for this note, add it to the score
                    if vel_type == "Velocity" and duration > 0:
                        new_note = Note(
                            time=current_time / 1000,
                            duration=duration / 1000,
                            pitch=pitch,
                            velocity=velocity,
                            ttype="Second"
                        )
                        score.tracks[0].notes.append(new_note)
                # A well constructed sequence should not usually raise an exception here.
                # However, this can happen with generated sequences, or sequences
                # that are chunks from a larger score
                except IndexError:
                    pass
        return score

    def _convert_sequence_to_tokseq(
            self,
            input_seq: list[int | str | list[int | str]] | np.ndarray,
    ) -> TokSequence | list[TokSequence]:
        """Convert a sequence of tokens/ids into a (list of) `TokSequence`."""
        # Deduce the type of data (ids/tokens/events)
        try:
            arg = ("ids", convert_ids_tensors_to_list(input_seq))
        except (AttributeError, ValueError, TypeError, IndexError):
            if isinstance(input_seq[0], str) or (
                    isinstance(input_seq[0], list) and isinstance(input_seq[0][0], str)
            ):
                arg = ("tokens", input_seq)
            else:  # list of Event, but unlikely
                arg = ("events", input_seq)
        # Deduce number of subscripts / dims
        num_io_dims = len(self.io_format)
        num_seq_dims = 1
        if len(arg[1]) > 0 and isinstance(arg[1][0], list):
            num_seq_dims += 1
            if len(arg[1][0]) > 0 and isinstance(arg[1][0][0], list):
                num_seq_dims += 1
            elif len(arg[1][0]) == 0 and num_seq_dims == num_io_dims - 1:
                # Special case where the sequence contains no tokens, we increment
                num_seq_dims += 1
        # Check the number of dimensions is good
        # In case of no one_token_stream and one dimension short --> unsqueeze
        if num_seq_dims == num_io_dims - 1:
            logger.warning(
                f"The input sequence has one dimension less than expected ("
                f"{num_seq_dims} instead of {num_io_dims}). It is being unsqueezed to "
                f"conform with the tokenizer's i/o format ({self.io_format})",
            )
            arg = (arg[0], [arg[1]])
        elif num_seq_dims != num_io_dims:
            msg = (
                f"The input sequence does not have the expected dimension "
                f"({num_seq_dims} instead of {num_io_dims})."
            )
            raise ValueError(msg)
        # Convert to TokSequence
        if num_io_dims == num_seq_dims:
            seq = []
            for obj in arg[1]:
                kwarg = {arg[0]: obj}
                seq.append(TokSequence(**kwarg))
        else:  # 1 subscript, one_token_stream and no multi-voc
            kwarg = {arg[0]: arg[1]}
            seq = TokSequence(**kwarg)
        return seq

    def decode(self, tokens: Union[TokSequence, list[TokSequence], list[int], np.ndarray], *_, **__) -> Score:
        """Detokenize sequences of tokens into a `symusic.Score` object"""
        # Coerce inputs to TokSequence
        if not isinstance(tokens, (TokSequence, list)) or (
                isinstance(tokens, list)
                and any(not isinstance(seq, TokSequence) for seq in tokens)
        ):
            tokens = self._convert_sequence_to_tokseq(tokens)

        # Preprocess TokSequence(s)
        if isinstance(tokens, TokSequence):
            self.complete_sequence(tokens)
        else:
            for seq in tokens:
                self.complete_sequence(seq)

        # Convert toksequence to score
        score = self._tokens_to_score(tokens[0].tokens)
        # Skipping over pedals
        # Set default tempo and time signature at tick 0 if not present
        if len(score.tempos) == 0 or score.tempos[0].time != 0:
            score.tempos.insert(0, Tempo(0, utils.TEMPO, ttype="Second"))
        if len(score.time_signatures) == 0 or score.time_signatures[0].time != 0:
            score.time_signatures.insert(
                0, TimeSignature(0, numerator=utils.TIME_SIGNATURE, denominator=4, ttype="Second")
            )
        return score

    def __call__(self, obj: Union[Score, list[int], np.ndarray], *args, **kwargs):
        """Tokenize a music file, or decode tokens into a `symusic.Score`"""
        # Tokenize `Score`
        if isinstance(obj, Score):
            return self.encode(obj, *args, **kwargs)
        # Path provided: Encode/decode a file
        if isinstance(obj, (str, Path)):
            obj = Path(obj)
            # Tokens
            if obj.suffix == "json":
                raise NotImplementedError
            return self.encode(obj, *args, **kwargs)
        # Decode tokens: may be a TokSequence, numpy array, or tensor
        return self.decode(obj, *args, **kwargs)

    @property
    def len(self) -> int:
        """Alias for __len__"""
        return len(self)

    def __len__(self) -> int:
        """Return the length of the vocabulary, as an integer"""
        return len(self.vocab)

    def __repr__(self):
        """Return the representation of the tokenizer, indicating its vocab size and i/o"""
        return f"{len(self)} tokens with {self.io_format} io format, not trained"

    def __getitem__(self, item: Union[int, str, tuple[int, Union[int, str]]]) -> Union[str, int, list[int]]:
        """Convert a token (int) to an event (str) or vica-versa"""
        if isinstance(item, tuple):
            raise NotImplementedError
        return self.__get_from_voc(item)

    def __get_from_voc(self, item: Union[int, str]) -> Union[int, str]:
        """Get an element from the vocabulary, handling id (int) <--> token (str) methods"""
        if isinstance(item, str):
            return self._vocab_base[item]
        else:
            return self.__vocab_base_inv[item]

    def __eq__(self, other) -> bool:
        """Check that two tokenizers are identical by comparing their vocabulary and configuration"""
        if not isinstance(other, type(self)):
            return False
        return self._vocab_base == other._vocab_base

    @property
    def vocab(self):
        return self._vocab_base

    def train(self, *args, **kwargs):
        raise NotImplementedError("`.train` is not implemented for this custom TSD-style tokenizer")

    @property
    def vocab_size(self) -> int:
        """Alias for __len__"""
        return len(self)

    @property
    def pad_token_id(self):
        """Return the ID of the padding token (``PAD_None``), usually 0."""
        return self._vocab_base["PAD_None"]

    @property
    def is_trained(self) -> bool:
        return False


class CustomTokTrainingIterator(TokTrainingIterator):
    """Modifies miditok.TokTrainingIterator to use our custom Score loading and preprocessing functions"""

    def __init__(
            self,
            tokenizer: MusicTokenizer,
            files_paths: Sequence[Path],
            tracks_idx_random_ratio_range: Union[tuple[float, float], None] = None,
            bars_idx_random_ratio_range: Union[tuple[float, float], None] = None,
    ):
        super().__init__(tokenizer, files_paths, tracks_idx_random_ratio_range, bars_idx_random_ratio_range)
        self.condition_tokens = [
            ct for ct in tokenizer.vocab if ct.startswith(("GENRES", "PIANIST", "TEMPO", "TIMESIGNATURE"))
        ]

    def load_file(self, path: Path) -> list[str]:
        """Load a file and preprocess with our custom functions, then convert to a byte representation"""
        # Load the score using our custom loading function
        try:
            score = load_score(path)
        except SCORE_LOADING_EXCEPTION:
            return []
        # Apply our own preprocessing to the score
        score = preprocess_score(score)
        # Stuff below is copied from MIDITok.tokenizer_training_iterator.TokTrainingIterator unless indicated
        # Preprocess first to already have the appropriate tracks idx in case of deletes
        score = self.tokenizer.preprocess_score(score)
        # Randomly create attribute controls indexes
        ac_indexes = None
        if (
                len(self.tracks_idx_random_ratio_range) > 0
                or len(self.bars_idx_random_ratio_range) > 0
        ):
            ac_indexes = create_random_ac_indexes(
                score,
                self.tokenizer.attribute_controls,
                self.tracks_idx_random_ratio_range,
                self.bars_idx_random_ratio_range,
            )
        # Tokenize the file
        # REMOVED: stuff to do with MMM tokenization
        tokseq = self.tokenizer(
            score,
            encode_ids=False,
            no_preprocess_score=True,
            attribute_controls_indexes=ac_indexes,
        )
        # ADDED: check that no tokens are in the list of condition tokens
        for tok in tokseq[0].tokens:
            assert tok not in self.condition_tokens

        # REMOVED: splitting IDs (we don't want to do this ever)
        # Convert ids to bytes for training
        if isinstance(tokseq, TokSequence):
            token_ids = tokseq.ids
        else:
            token_ids = [seq.ids for seq in tokseq]
        bytes_ = self.tokenizer._ids_to_bytes(token_ids, as_one_str=True)
        if isinstance(bytes_, str):
            bytes_ = [bytes_]
        return bytes_


def add_pianists_to_vocab(tokenizer) -> None:
    """Adds all valid pianists on all tracks to the tokenizer"""
    for pianist in INCLUDE["pianist"]:
        # Add the tokenizer prefix here
        with_prefix = f'PIANIST_{utils.remove_punctuation(pianist).replace(" ", "")}'
        if with_prefix not in tokenizer.vocab:
            tokenizer.add_to_vocab(with_prefix, special_token=False)


def add_genres_to_vocab(tokenizer: MusicTokenizer) -> None:
    """Adds all valid genres for all tracks and artists to the tokenizer"""
    for genre in list(INCLUDE["genres"]):
        with_prefix = f'GENRES_{utils.remove_punctuation(genre).replace(" ", "")}'
        if with_prefix not in tokenizer.vocab:
            tokenizer.add_to_vocab(with_prefix, special_token=False)


def add_timesignatures_to_vocab(tokenizer: MusicTokenizer, time_signatures: list[int]) -> None:
    """Given a list of time signatures, add these to the vocabulary as custom tokens (shouldn't be used in decoding)"""
    for time_signature in time_signatures:
        tok_id = f'TIMESIGNATURECUSTOM_{time_signature}4'
        if tok_id not in tokenizer.vocab:
            tokenizer.add_to_vocab(tok_id, special_token=False)


def add_recording_years_to_vocab(
        tokenizer: MusicTokenizer, min_year: int = 1945, max_year: int = 2025, step: int = 5
) -> None:
    """Adds year tokens to vocabulary: linearly spaced between min and max years according to step"""
    year_range = range(min_year, max_year + 1, step)
    for year in year_range:
        tok_id = f'RECORDINGYEAR_{year}'
        if tok_id not in tokenizer.vocab:
            tokenizer.add_to_vocab(tok_id, special_token=False)


def add_tempos_to_vocab(tokenizer: MusicTokenizer, min_tempo: int, n_tempos: int = 30, factor: float = 1.05) -> None:
    """Add tempo tokens to vocabulary using geometric distribution"""
    # Create the geometric distribution
    tempo_range = [min_tempo]
    for _ in range(n_tempos - 1):
        tempo_range.append(round(tempo_range[-1] * factor))
    # Add the tokens
    for tempo in tempo_range:
        tok_id = f'TEMPOCUSTOM_{tempo}'
        if tok_id not in tokenizer.vocab:
            tokenizer.add_to_vocab(tok_id, special_token=False)


def get_tokenizer_class_from_string(tokenizer_type: str):
    """Given a string, return the correct tokenizer class"""
    valids = ["remi", "midilike", "tsd", "structured", "pertok", "custom-tsd"]
    tokenizer_type = tokenizer_type.lower()
    if tokenizer_type == "remi":
        return REMI
    elif tokenizer_type == "midilike":
        return MIDILike
    elif tokenizer_type == "tsd":
        return TSD
    elif tokenizer_type == "structured":
        return Structured
    elif tokenizer_type == "pertok":
        return PerTok
    elif tokenizer_type == "custom-tsd":
        return CustomTSD
    else:
        raise ValueError(f'`tokenizer_type` must be one of {", ".join(valids)} but got {tokenizer_type}')


def fix_pertok_microtiming_bug(tokenizer: PerTok) -> None:
    """Fixes https://github.com/Natooz/MidiTok/issues/227 by setting a missing attribute for the PerTok tokenizer"""
    if not hasattr(tokenizer, "microtiming_tick_values"):
        # This just copies the code from miditok.tokenizers.pertok, line 138
        mt_bins = tokenizer.config.additional_params["num_microtiming_bins"]
        tokenizer.microtiming_tick_values = np.linspace(
            -tokenizer.max_mt_shift,
            tokenizer.max_mt_shift,
            mt_bins + 1,
            dtype=np.intc
        )
        assert hasattr(tokenizer, "microtiming_tick_values")


def load_tokenizer(**kwargs) -> MusicTokenizer:
    # Get the name of the tokenizer from the config dictionary
    tokenizer_method = kwargs.get("tokenizer_str", DEFAULT_TOKENIZER_CLASS)
    # Try and load a trained tokenizer
    tokenizer_path = kwargs.get("tokenizer_path", False)
    if os.path.isfile(tokenizer_path):
        logger.debug(f'Initialising tokenizer tupe {tokenizer_method} from path {tokenizer_path}')
        tokenizer = get_tokenizer_class_from_string(tokenizer_method)(params=tokenizer_path)
    # Otherwise, create the tokenizer from scratch
    else:
        tokenizer_kws = kwargs.get("tokenizer_kws", DEFAULT_TOKENIZER_CONFIG)
        # Add in any missing parameters with defaults
        tokenizer_kws = utils.update_dictionary(tokenizer_kws, DEFAULT_TOKENIZER_CONFIG)
        logger.debug(f'Initialising tokenizer type {tokenizer_method} with params {tokenizer_kws}')
        # Create the tokenizer configuration + tokenizer
        cfg = CustomTokenizerConfig(**tokenizer_kws)
        tokenizer = get_tokenizer_class_from_string(tokenizer_method)(cfg)
    logger.debug(f'... got tokenizer: {tokenizer}')
    # We need to set this attribute to make decoding tokens easier
    #  At this point, the mapping just goes from token IDX -> [token IDX]
    #  However, if we train the tokenizer, we'll update it to go BPE token IDX -> [token1 IDX, token2 IDX]
    #  By setting it here, we ensure compatibility between trained + non-trained tokenizers when calculating
    #  evaluation metrics e.g. negative log-likelihood loss, accuracy scores etc.
    setattr(tokenizer, "bpe_token_mapping", {v: [v] for v in tokenizer.vocab.values()})
    return tokenizer


def train_tokenizer(tokenizer: MusicTokenizer, files_paths: list[str], **kwargs) -> None:
    """Trains a tokenizer given kwargs using our custom iterator class"""
    # We don't need to train a tokenizer if it's already been trained!
    if tokenizer.is_trained:
        logger.warning(f'... tried to train a tokenizer that has already been trained, skipping')
    else:
        # Get the parameters again from the dictionary
        training_method = kwargs.get("training_method", DEFAULT_TRAINING_METHOD)
        vocab_size = kwargs.get("vocab_size", DEFAULT_VOCAB_SIZE)

        # If we try to train with a smaller vocab size than the model currently has
        if vocab_size <= tokenizer.vocab_size:
            # Skip over training and just return without updating anything
            logger.warning(f'... tried to train a tokenizer with a smaller vocabulary size than it '
                           f'has already ({vocab_size} vs. {tokenizer.vocab_size}), skipping')
            return

        logger.debug(f'... training tokenizer with method {training_method}, vocab size {vocab_size}')
        # We need to train with our custom iterator so that we use our custom score loading + preprocessing functions
        utils.validate_paths(files_paths, expected_extension=".mid")
        tti = CustomTokTrainingIterator(tokenizer, files_paths)
        tokenizer.train(vocab_size=vocab_size, model=training_method, iterator=tti)
        logger.debug(f'... training finished: {tokenizer}')
    # Now, we update our token mapping to go BPE token1 IDX -> [token1 IDX, token2 IDX], ...
    #  This will trigger if the tokenizer has already been trained BEFORE calling this function, too
    bpe_token_mapping = {
        tokenizer.vocab_model[byt]: [tokenizer[t] for t in token_list]
        for byt, token_list in tqdm(tokenizer._vocab_learned_bytes_to_tokens.items(), desc="Creating token mapping...")
    }
    setattr(tokenizer, "bpe_token_mapping", bpe_token_mapping)


if __name__ == "__main__":
    from time import time

    tokfactory = CustomTSD(CustomTokenizerConfig(time_factor=1.03, time_range=(0.01, 2.5)))
    all_times = []
    to_process = os.listdir(os.path.join(utils.get_project_root(), "data/raw/pijama"))[300:400]
    with utils.timer("encode custom"):
        for i in tqdm(to_process, desc="Encoding custom"):
            midi = os.path.join(utils.get_project_root(), "data/raw/pijama", i, "piano_midi.mid")
            raw = preprocess_score(load_score(midi, as_seconds=True))
            start = time()
            enc = tokfactory.encode(os.path.join(utils.get_project_root(), midi))
            all_times.append(time() - start)
    print(np.mean(all_times))

    all_times = []
    tokfactory_vanilla = load_tokenizer(tokenizer_str="tsd")
    with utils.timer("encode vanilla"):
        for i in tqdm(to_process, desc="Encoding vanilla"):
            midi = os.path.join(utils.get_project_root(), "data/raw/pijama", i, "piano_midi.mid")
            raw = preprocess_score(load_score(midi, as_seconds=True))
            start = time()
            enc = tokfactory_vanilla.encode(os.path.join(utils.get_project_root(), midi))
            all_times.append(time() - start)
    print(np.mean(all_times))
