#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Files for loading and preprocessing score objects"""

from copy import deepcopy

from symusic import Score, Note, Track, Tempo, TimeSignature
from symusic.core import Second

from jazz_style_conditioned_generation import utils

OVERLAP_MILLISECONDS = 0  # If two notes with the same pitch have less than this offset-onset time, they will be merged
MIN_DURATION_MILLISECONDS = 10  # We remove notes that have a duration of less than this value
MAX_DURATION_MILLISECONDS = 5000  # we cap notes with a duration longer than this to this value
QUANTIZE_MILLISECONDS = 10  # We quantize notes to the nearest 10 ms

TINY = 1e-4


def get_notes_from_score(score: Score) -> list[Note]:
    """Given a score that may have multiple tracks, we only want to get the notes from the correct (piano) track"""
    # If we somehow have more than one track (occasionally happens in the bushgrafts corpus)
    if len(score.tracks) > 1:
        # Get all the piano tracks
        is_piano = [p for p in score.tracks if p.program == utils.MIDI_PIANO_PROGRAM]
        # Keep the one with the most notes
        desired_track = max(is_piano, key=lambda x: len(x.notes))
        return sorted(desired_track.notes, key=lambda x: x.start)
    # Otherwise, we can just grab the track directly
    else:
        return sorted(score.tracks[0].notes, key=lambda x: x.start)


def load_score(filepath: str, as_seconds: bool = False) -> Score:
    """Loads a MIDI file and resamples such that 1 tick == 1 millisecond in real time"""
    # Load as a symusic object with time sampled in seconds and sort the notes by onset time
    score_as_secs = Score(filepath, ttype="Second")  # this preserves tempo, time signature information etc.
    secs_notes = get_notes_from_score(score_as_secs)
    # Create an EMPTY symusic object with time sampled in ticks
    ttype = "Second" if as_seconds else "Tick"
    score_obj = Score(ttype=ttype).resample(tpq=utils.TICKS_PER_QUARTER)
    # Need to convert back to seconds, as resample automatically converts to ticks
    if as_seconds:
        score_obj = score_obj.to(ttype)
    # Add in required attributes: tracks, tempo, time signatures
    score_obj.tracks = [Track(program=0, ttype=ttype)]
    score_obj.tempos = [Tempo(time=0, qpm=utils.TEMPO, ttype=ttype)]
    score_obj.time_signatures = [
        TimeSignature(time=0, numerator=utils.TIME_SIGNATURE, denominator=4, ttype=ttype)
    ]
    if as_seconds:
        score_obj.tracks[0].notes = secs_notes
        return score_obj
    # Iterate over the notes sampled in seconds
    newnotes = []
    for n in secs_notes:
        # Convert their time into ticks/milliseconds
        new_start = utils.base_round(n.time * 1000, 10)  # rounding to the nearest 10 milliseconds
        new_duration = utils.base_round(n.duration * 1000, 10)  # rounding to the nearest 10 milliseconds
        newnote = Note(
            time=new_start,
            duration=new_duration,
            pitch=n.pitch,
            velocity=n.velocity,
            ttype="Tick"
        )
        newnotes.append(newnote)
    # Set the notes correctly
    score_obj.tracks[0].notes = newnotes
    return score_obj


def remove_short_notes(note_list: list[Note], min_duration_milliseconds: int = MIN_DURATION_MILLISECONDS) -> list[Note]:
    """Removes symusic.Note objects with a duration of less than min_duration_milliseconds from a list of Notes"""
    newnotes = []
    for note in note_list:
        # Notes with a duration this short are transcription errors usually
        if note.duration >= min_duration_milliseconds:
            newnotes.append(note)
    return newnotes


def cap_long_notes(note_list: list[Note], max_duration_milliseconds: int = MAX_DURATION_MILLISECONDS):
    """Cap symusic.Note objects with a duration of longer than max_duration_milliseconds to max_duration_milliseconds"""
    newnotes = []
    for note in note_list:
        # Notes with a duration this long will be capped to the maximum
        if note.duration >= max_duration_milliseconds:
            note.duration = max_duration_milliseconds
        newnotes.append(note)
    return newnotes


def merge_repeated_notes(note_list: list[Note], overlap_milliseconds: int = OVERLAP_MILLISECONDS) -> list[Note]:
    """Merge successive notes at the same pitch with an offset-onset time < overlap_milliseconds to a single note"""
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
                if overlap < overlap_milliseconds:
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


def remove_overlap(note_list: list[Note]) -> list[Note]:
    """
    If the offset-onset time for two successive notes at the same pitch overlap,
    set the offset of the first note to the onset of the second
    """
    newnotes = []
    # Padding to separate overlapping offsets/onsets: 10 ms
    pad = 0.01 if isinstance(note_list[0].ttype, Second) else 10
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
            # Iterate over successive pairs of notes (note1, note2), (note2, note3)...
            for note_idx in range(len(notes_sorted) - 1):
                # Unpack to get desired notes
                note1 = notes_sorted[note_idx]
                note2 = notes_sorted[note_idx + 1]
                # If the earlier note overlaps the later one
                if note1.time + note1.duration > note2.time:
                    # Set its offset time to be the onset time of the subsequent note, minus padding
                    note1.duration = note2.time - note1.time - pad
                newnotes.append(note1)  # add only once
            # After the loop, add the last note
            newnotes.append(notes_sorted[-1])
    return newnotes


def remove_out_of_range_notes(note_list: list[Note]) -> list[Note]:
    """Remove notes from a list that are outside the range of the piano keyboard"""
    note_list_ = deepcopy(note_list)
    return [n for n in note_list_ if utils.MIDI_OFFSET <= n.pitch < utils.MIDI_OFFSET + utils.PIANO_KEYS]


def note_list_to_score(note_list: list[Note], ticks_per_quarter: int, ttype: str = "tick") -> Score:
    """Converts a list of symusic.Note objects to a single symusic.Score"""
    # This API is fairly similar to pretty_midi
    newscore = Score(ttype=ttype)
    newscore.ticks_per_quarter = ticks_per_quarter
    newscore.tracks = [Track(program=utils.MIDI_PIANO_PROGRAM, ttype=ttype)]
    newscore.tracks[0].notes = note_list
    return newscore


def remove_duplicate_notes(note_list: list[Note]):
    """Removes duplicates from a list of notes, based on pitch, onset, duration, and velocity"""
    seen = set()
    unique_notes = []
    # Iterate over all the notes
    for note in note_list:
        note_key = (note.pitch, note.time, note.duration, note.velocity)
        # Add unique notes (based on pitch, time, duration, and velocity) to the list
        if note_key not in seen:
            seen.add(note_key)
            unique_notes.append(note)
    return unique_notes


def align_to_start(notes: list[Note]) -> list[Note]:
    """Aligns a list of notes such that the earliest onset time == 0 seconds"""
    # Prevents `min arg is an empty sequence` when no notes present
    if len(notes) == 0:
        return []
    # Get the earliest note by onset time: break ties by using offset time
    first_note = min(notes, key=lambda x: (x.start, x.end))
    newnotes = []
    # Iterate over all the notes in the list
    for note in notes:
        # Create a new note where the time is shifted to align the note on 0
        newnote = Note(
            time=note.time - first_note.time,
            duration=note.duration,  # maintain as before
            pitch=note.pitch,  # maintain as before
            velocity=note.velocity,  # maintain as before
            ttype=note.ttype  # maintain as before
        )
        newnotes.append(newnote)
    # Return the new list of notes
    return newnotes


def quantize_notes(notes: list[Note], quantize_resolution: float) -> list[Note]:
    """Quantize note onset and duration times according to a given resolution"""
    newnotes = []
    for note in notes:
        # Quantize note start and duration times
        start = round(note.time / quantize_resolution) * quantize_resolution
        duration = round(note.duration / quantize_resolution) * quantize_resolution
        # Skip over notes that are smaller than our quantized resolution
        if duration + TINY < quantize_resolution:  # adding a small epsilon value should help floating-point precision
            continue
        # For time in ticks, the start and duration value must be integers, not floats
        if not isinstance(note.ttype, Second):
            start = int(round(start))
            duration = int(round(duration))
        # Create the new note object and append it to the list
        newnote = Note(
            pitch=note.pitch,
            velocity=note.velocity,
            ttype=note.ttype,
            time=start,
            duration=duration
        )
        newnotes.append(newnote)
    return newnotes


def preprocess_score(
        score: Score,
        min_duration_milliseconds: int = MIN_DURATION_MILLISECONDS,
        overlap_milliseconds: int = OVERLAP_MILLISECONDS,
        max_duration_milliseconds: int = MAX_DURATION_MILLISECONDS,
        quantize_milliseconds: int = QUANTIZE_MILLISECONDS
) -> Score:
    """Applies our own preprocessing to a Score object: removes short notes, merges duplicates"""
    # Scores in seconds, not "hacked" ticks: we need to convert the times from milliseconds to seconds
    if isinstance(score.ttype, Second):
        min_duration_milliseconds /= 1000
        overlap_milliseconds /= 1000
        max_duration_milliseconds /= 1000
        quantize_milliseconds /= 1000
    # Get the notes from the score
    score_ = deepcopy(score)
    note_list = get_notes_from_score(score_)
    # Remove notes that are outside the range of the piano keyboard
    validated_notes = remove_out_of_range_notes(note_list)
    # Cap notes with an exceptionally long duration to the max duration
    no_long_notes = cap_long_notes(validated_notes, max_duration_milliseconds=max_duration_milliseconds)
    # Quantize notes to the nearest 10 ms
    quantized_notes = quantize_notes(no_long_notes, quantize_milliseconds)
    # Remove any overlap between successive onset-offset times of the same note
    no_overlap_notes = remove_overlap(quantized_notes)
    # Remove notes with a very short duration
    no_short_notes = remove_short_notes(no_overlap_notes, min_duration_milliseconds=min_duration_milliseconds)
    # Align the notes such that the earliest onset time == 0
    aligned_notes = align_to_start(no_short_notes)
    # De-duplicate the list of notes
    deduped_notes = remove_duplicate_notes(aligned_notes)
    # Sort the notes by time, pitch, duration, and velocity (same order as MidiTok/Symusic default)
    deduped_notes.sort(key=lambda n: (n.time, n.pitch, n.duration, n.velocity))
    # Finally, convert everything back to a Score object that can be passed to our tokenizer
    score_.tracks[0].notes = deduped_notes
    return score_


if __name__ == "__main__":
    import os

    from jazz_style_conditioned_generation.data.tokenizer import (
        DEFAULT_TOKENIZER_CONFIG, CustomTSD, CustomTokenizerConfig
    )

    file = "data/pretraining/atepp/handelgf-harpsichordsuiteingminor-richters-2008-d64629e4/piano_midi.mid"
    mf = os.path.join(utils.get_project_root(), file)
    loaded = load_score(mf, as_seconds=True)

    tokenizer = CustomTSD(CustomTokenizerConfig(**DEFAULT_TOKENIZER_CONFIG, time_range=(0.01, 1.0), time_factor=1.0))

    preproc = preprocess_score(loaded)
    enc = tokenizer.encode(preproc)
    dec = tokenizer.decode(enc)
    dec.dump_midi(os.path.join(utils.get_project_root(), f"preproc.mid"))
