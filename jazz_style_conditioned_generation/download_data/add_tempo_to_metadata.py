#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Add tempo and time signature information to metadata for tracks that do not have this already"""

import os
import re
from pathlib import Path

import numpy as np
from scipy.stats import mode
from symusic import Score

from jazz_style_conditioned_generation import utils


def process_jja(jja_dir: str = "data/raw/jja") -> None:
    """Process tempi and time signature for JJA dataset: this can be obtained from the filename"""
    # This just gets the filepaths for the metadata files for each recording
    jja = utils.get_data_files_with_ext(jja_dir, ext="**/*_tivo.json")
    # This gets the directory name (one per track)
    jja_dirnames = [str(Path(i).parent).split(os.path.sep)[-1] for i in jja]
    # Extract the tempo from the filename
    jja_tempos = [float(re.search(r'_(\d+)-bass', j).group(1)) for j in jja_dirnames]
    assert len(jja_tempos) == len(jja_dirnames)
    # Iterate over all the metadata files + associated tempi
    for metadata_file, bpm in zip(jja, jja_tempos):
        metadata_read = utils.read_json_cached(metadata_file)
        # Add the tempo in if we haven't done so already
        if "tempo" not in metadata_read.keys():
            metadata_read["tempo"] = bpm
        # All the recordings for this dataset are in 4/4, so we don't need to do any processing
        if "time_signature" not in metadata_read.keys():
            metadata_read["time_signature"] = 4
        # Dump the JSON, overwriting the previous dictionary
        utils.write_json(metadata_read, metadata_file)


def process_bushgrafts(bushgrafts_dir: str = "data/raw/bushgrafts"):
    """Process tempi and time signature for Bushgrafts dataset: this can be obtained from the MIDI file"""
    bg_midis = utils.get_data_files_with_ext(bushgrafts_dir, ext="**/*.mid")
    for midi in bg_midis:
        # Load the score in
        loaded = Score(midi, ttype="tick")
        # Get the tempi and average
        tempi = [float(i.qpm) for i in loaded.tempos]
        mean_tempi = float(np.nanmean(tempi))
        # Get the time signature numerators and average
        time_signatures = [int(i.numerator) for i in loaded.time_signatures]
        # Get the most frequently occurring time signature
        modal_ts = int(mode(time_signatures)[0])
        # Load up the metadata
        metadata_fp = midi.replace("piano_midi.mid", "metadata_tivo.json")
        metadata_loaded = utils.read_json_cached(metadata_fp)
        # Add in the parameters if we haven't done so already
        if "tempo" not in metadata_loaded.keys():
            metadata_loaded["tempo"] = mean_tempi
        if "time_signature" not in metadata_loaded.keys():
            metadata_loaded["time_signature"] = modal_ts
        # Dump the JSON, overwriting the previous dictionary
        utils.write_json(metadata_loaded, metadata_fp)


def main():
    process_jja()
    process_bushgrafts()


if __name__ == "__main__":
    main()
