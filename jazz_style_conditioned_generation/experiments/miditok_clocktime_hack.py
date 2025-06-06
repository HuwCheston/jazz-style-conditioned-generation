#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""MIDITok expects time to be expressed in terms of ticks, beats, and bars. If a `symusic.Score` object (used by
MIDITok to tokenize MIDI files) does not have this information, or we want to express time in terms of seconds (clock
time), using MIDITok can be difficult.

To solve this problem, we can hack MIDITok by creating a new Score object that makes ticks directly equivalent to
milliseconds (i.e., 1 tick == 1 millisecond).
"""

import os
import random
from time import time

from miditok import MIDILike, TokenizerConfig
from pretty_midi import Note as PMNote
from pretty_midi import PrettyMIDI, Instrument
from pretty_midi import TimeSignature as PMTimeSignature
from symusic import Tempo, TimeSignature, Score, Note, Track
from tqdm import tqdm

# With these values, 1 bar of 2 beats == 1 second
DESIRED_TEMPO = 120
DESIRED_TPQ = 500
DESIRED_TS = 2
RUNS = 10
# Create the tokenizer
TOKENIZER = MIDILike(
    TokenizerConfig(
        # This means that we will have exactly 100 evenly-spaced tokens per "bar"
        beat_res={(0, DESIRED_TS): 100 // DESIRED_TS}
    )
)


def base_round(x: float, base: int = 10) -> int:
    """Rounds a number to the nearest base"""
    return int(base * round(float(x) / base))


def symusic_notes_to_string(sy_score: Score) -> str:
    """Converts symusic notes to a nicely-formatted string"""
    return ", ".join(f'start: {n.start}, end: {n.start + n.duration}' for n in sy_score.tracks[0].notes)


def load_score(filepath: str) -> Score:
    """Loads a MIDI file and resamples such that 1 tick == 1 millisecond in real time"""
    # Load as a symusic object with time sampled in seconds and sort the notes by onset time
    score_as_secs = Score(filepath, ttype="Second")  # this preserves tempo, time signature information etc.
    secs_notes = sorted(score_as_secs.tracks[0].notes.copy(), key=lambda x: x.start)
    # Create an EMPTY symusic object with time sampled in ticks
    score_as_ticks = Score(ttype="Tick").resample(tpq=DESIRED_TPQ)
    # Add in required attributes: tracks, tempo, time signatures
    score_as_ticks.tracks = [Track(program=0, ttype="Tick")]
    score_as_ticks.tempos = [Tempo(time=0, qpm=DESIRED_TEMPO, ttype="Tick")]
    score_as_ticks.time_signatures = [TimeSignature(time=0, numerator=DESIRED_TS, denominator=4, ttype="Tick")]
    # Iterate over the notes sampled in seconds
    newnotes = []
    for n in secs_notes:
        # Convert their time into ticks/milliseconds
        new_start = base_round(n.time * 1000, 10)  # rounding to the nearest 10 milliseconds
        new_duration = base_round(n.duration * 1000, 10)  # rounding to the nearest 10 milliseconds
        newnote = Note(
            time=new_start,
            duration=new_duration,
            pitch=n.pitch,
            velocity=n.velocity,
            ttype="Tick"
        )
        newnotes.append(newnote)
    # Set the notes correctly
    score_as_ticks.tracks[0].notes = newnotes
    return score_as_ticks


# The tokenizer vocab is a dictionary: we just want to get the TimeShift tokens
timeshifts = [m for m in TOKENIZER.vocab.keys() if "TimeShift" in m]
n_timeshifts = len(timeshifts)  # 100 tokens at evenly-spaced 10ms increments between 0 - 1 second
assert n_timeshifts == 100

# Get tokens corresponding to particular time values
token_10ms = timeshifts[0]
token_50ms = timeshifts[4]
token_100ms = timeshifts[9]
token_500ms = timeshifts[49]
token_1000ms = timeshifts[-1]
# Create some dummy pretty MIDI notes with standard durations
dummy_notes = [
    PMNote(start=0., end=1., pitch=80, velocity=50),  # 1 second duration
    PMNote(start=1., end=1.1, pitch=81, velocity=50),  # 100 millisecond duration
    PMNote(start=1.1, end=1.11, pitch=82, velocity=50),  # 10 millisecond duration
    PMNote(start=1.11, end=1.16, pitch=83, velocity=50),  # 50 millisecond duration
    PMNote(start=1.16, end=1.66, pitch=84, velocity=50),  # 500 millisecond duration
    PMNote(start=1.66, end=3.66, pitch=85, velocity=50),  # 2 second duration
    PMNote(start=3.66, end=5.67, pitch=86, velocity=50),  # 2 second + 10 millisecond duration
]
# Convert our dummy note stream into the expected tokens
expected_tokens = [
    'NoteOn_80', 'Velocity_51', token_1000ms, 'NoteOff_80',
    'NoteOn_81', 'Velocity_51', token_100ms, 'NoteOff_81',
    'NoteOn_82', 'Velocity_51', token_10ms, 'NoteOff_82',
    'NoteOn_83', 'Velocity_51', token_50ms, 'NoteOff_83',
    'NoteOn_84', 'Velocity_51', token_500ms, 'NoteOff_84',
    'NoteOn_85', 'Velocity_51', token_1000ms, token_1000ms, 'NoteOff_85',
    'NoteOn_86', 'Velocity_51', token_1000ms, token_1000ms, token_10ms, 'NoteOff_86',
]
print("Dummy note stream (seconds):", ", ".join(f'start: {n.start}, end: {n.end}' for n in dummy_notes))
print(f'Expected tokens: ', expected_tokens)
print("\n\n")

# TESTING WITH DUMMY MIDI FILES
for i in range(1, RUNS + 1):
    print(f"RUN {i}")
    # Create a fake MIDI file
    # Get a random tempo and time signature
    tempo = random.randint(100, 300)
    ts = random.randint(1, 8)
    tpq = random.randint(100, 1000)
    # Create a PrettyMIDI object at the desired resolution with a random tempo
    pm = PrettyMIDI(resolution=tpq, initial_tempo=tempo)
    pm.instruments = [Instrument(program=0)]
    # Add some notes in
    pm.instruments[0].notes = dummy_notes
    # Add some random time signature and tempo changes
    pm.time_signature_changes = [PMTimeSignature(ts, 4, 0)]
    # Sanity check the input
    assert pm.resolution == tpq
    assert pm.time_signature_changes[0].numerator == ts
    assert round(pm.get_tempo_changes()[1][0]) == tempo
    # Log everything for the input
    print(f"Input time signature: ", pm.time_signature_changes[0].numerator, "/ 4")
    print(f'Input resolution: ', pm.resolution)
    print(f'Input tempo: ', round(pm.get_tempo_changes()[1][0]))
    # Write our random midi file
    out_path = "temp_pm.mid"
    pm.write(out_path)

    # Log what the tokens/times would be without our modifications
    score_no_mod = Score(out_path, ttype="Tick")
    toks_no_mod = TOKENIZER.encode(score_no_mod)[0].tokens
    print(f'Input notes (ticks): ', symusic_notes_to_string(score_no_mod))
    print(f'Input tokens: ', toks_no_mod)

    # Load in as a symusic score object and apply our preprocessing to standardise tempo, time signature, & TPQ
    score = load_score(out_path)
    # Sanity check that the score is correct
    assert score.ticks_per_quarter == DESIRED_TPQ
    assert score.tempos[0].qpm == DESIRED_TEMPO
    assert score.time_signatures[0].numerator == DESIRED_TS
    # Tokenize the output
    toks = TOKENIZER.encode(score)[0].tokens
    # Log everything
    print(f"Output time signature: ", score.time_signatures[0].numerator, "/ 4")
    print(f'Output resolution: ', score.ticks_per_quarter)
    print(f'Output tempo: ', round(score.tempos[0].qpm))
    print(f'Output notes (ticks): ', symusic_notes_to_string(score))
    print(f'Output tokens: ', toks)
    # Sanity check that the tokens are correct
    only_timeshifts = [t for t in toks if "TimeShift" in t]
    assert only_timeshifts[0] == token_1000ms
    assert only_timeshifts[1] == token_100ms
    assert only_timeshifts[2] == token_10ms
    assert only_timeshifts[3] == token_50ms
    assert only_timeshifts[4] == token_500ms
    assert toks == expected_tokens
    # Cleaning up
    print("\n\n")
    os.remove(out_path)

# TESTING WITH ACTUAL MIDI FILES
# You can replace these files to point to actual MIDIs on your system
files = [
    "../../tests/test_resources/test_midi1/piano_midi.mid",
    "../../tests/test_resources/test_midi2/piano_midi.mid",
    "../../tests/test_resources/test_midi3/piano_midi.mid",
    "../../tests/test_resources/test_midi_jja1/piano_midi.mid",
    "../../tests/test_resources/test_midi_bushgrafts1/piano_midi.mid",
    "../../tests/test_resources/test_midi_bushgrafts2/piano_midi.mid",
    "../../tests/test_resources/test_midi_bushgrafts3/piano_midi.mid",
]
for file in files:
    print("Loading file: ", file)
    # Load with Symusic and ttype=="Second" to get actual times, based on tempo, time signature, and resolution of input
    pm_load = Score(file, ttype="Second")
    print(f"Input time signatures: ", ", ".join(str(i.numerator) for i in pm_load.time_signatures))
    print(f'Input resolution: ', pm_load.ticks_per_quarter)
    print(f'Input tempos: ', ", ".join(str(round(i.qpm)) for i in pm_load.tempos))
    # Load with our custom symusic function
    sm_load = load_score(file)
    print(f"Output time signatures: ", ", ".join(str(i.numerator) for i in sm_load.time_signatures))
    print(f'Output resolution: ', sm_load.ticks_per_quarter)
    print(f'Output tempo: ', ", ".join(str(round(i.qpm)) for i in sm_load.tempos))
    # Get the notes from both files and sort by onset time (not sure if this is necessary)
    pm_notes = sorted(pm_load.tracks[0].notes, key=lambda x: x.start)
    sm_notes = sorted(sm_load.tracks[0].notes, key=lambda x: x.start)
    # Number of notes should be the same
    assert len(pm_notes) == len(sm_notes)
    # Start times for notes should be directly equivalent
    for pm, sm in zip(pm_notes, sm_notes):
        assert base_round(pm.start * 1000, 10) == sm.time  # make sure to round to nearest 10ms!
        assert base_round(pm.duration * 1000, 10) == sm.duration
    print("\n\n")

# PROFILING WITH ACTUAL MIDI FILES
res = []
for _ in tqdm(range(RUNS), desc="Profiling load_score..."):
    start = time()
    loaded = load_score(files[0])
    res.append(time() - start)
print("Average time to load file: ", round(sum(res) / len(res), 3), "seconds")
