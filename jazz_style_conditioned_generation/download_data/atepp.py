#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Downloads and preprocesses MIDI files from https://github.com/tangjjbetsy/ATEPP/blob/master/disclaimer.md"""

import os
import random
import shutil
import subprocess
from secrets import token_hex

import gdown
import pandas as pd
from tqdm import tqdm

from jazz_style_conditioned_generation import utils
from jazz_style_conditioned_generation.preprocessing.splits import check_all_splits_unique, SPLIT_DIR

ATEPP_URL = "https://drive.google.com/uc?id=1Df2KUdqvXtgvhvzx2D10YaQRbpP23rOS"
DESTINATION = os.path.join(utils.get_project_root(), "data/pretraining/atepp")
PRETRAIN_SPLIT_DIR = os.path.join(SPLIT_DIR, "pretraining")


def download_file_from_google_drive() -> str:
    zip_path = os.path.join(DESTINATION, "temp.zip")
    gdown.download(ATEPP_URL, zip_path)
    return zip_path


def format_tracks() -> list[dict]:
    # Read the metadata in as a pandas dataframe
    metadata = pd.read_csv(os.path.join(DESTINATION, "ATEPP-metadata-1.2.csv"))
    all_tracks = []

    # Iterate over all tracks in the directory
    for direc, subdir, file in tqdm(os.walk(os.path.join(DESTINATION, "ATEPP-1.2"))):
        for f in file:
            if f.endswith(".mid"):
                # Formatting composer name
                composer = direc.split("ATEPP-1.2")[1].split(os.path.sep)[1]
                splitted = composer.lower().split("_")
                composer_fmt = utils.remove_punctuation(f'{splitted[-1]}{"".join(s[0] for s in splitted[:-1])}')

                # Formatting piece name
                piece_name = "_".join(direc.split("ATEPP-1.2")[1].split(os.path.sep)[2:])
                piece_splitted = piece_name.lower().split("_")
                piece_fmt = utils.remove_punctuation("".join(p for p in piece_splitted[:5]))

                # Get metadata for this performance
                row = metadata[metadata["perf_id"] == f.replace(".mid", "").split("-")[0]]
                if len(row) != 1:
                    continue
                row = row.iloc[0]

                # Skip over performances that are repeats
                if row["repetition"] == "repetition":
                    continue

                # Get recording year from the metadata
                recording_year = row["album_date"].split("-")[0]

                # Get artist from the metadata
                artist = row["artist"]
                artist_split = artist.lower().split(" ")
                artist_fmt = utils.remove_punctuation(f'{artist_split[-1]}{"".join(s[0] for s in artist_split[:-1])}')

                # Create folder name
                hasher = token_hex(4)
                folder_name = f'{composer_fmt}-{piece_fmt}-{artist_fmt}-{recording_year}-{hasher}'

                # Create metadata as a JSON
                track_metadata = dict(
                    track_name=row["track"],
                    album_name=row["album"],
                    recording_year=int(recording_year),
                    bandleader=row["composer"],
                    pianist=row["artist"],
                    mbz_id=f"{hasher}-{token_hex(2)}-{token_hex(2)}-{token_hex(2)}-{token_hex(6)}",
                    fname=folder_name,
                    _composition_id=int(row["composition_id"]),
                    _artist_id=int(row["artist_id"]),
                    _performance_id=int(row["perf_id"].lstrip())
                )

                # Create the folder
                folder_joined = os.path.join(DESTINATION, folder_name)
                os.makedirs(folder_joined, exist_ok=True)

                # Copy the MIDI
                os.rename(os.path.join(direc, f), os.path.join(folder_joined, "piano_midi.mid"))

                # Save the metadata
                utils.write_json(track_metadata, os.path.join(folder_joined, "metadata_tivo.json"))

                # Append to the list
                all_tracks.append(track_metadata)
    return all_tracks


def create_data_splits(track_metadatas: list[dict]):
    metadata_df = pd.DataFrame(track_metadatas)

    # Get the IDs and total number of unique compositions
    unique_compositions = metadata_df["_composition_id"].unique()
    nunique_compositions = len(unique_compositions)

    # Get the number of compositions that should be in each split
    n_train_comp = round(nunique_compositions * 0.9)
    # n_test_comp = round(nunique_compositions * 0.1)
    n_valid_comp = round(nunique_compositions * 0.1)

    # Add any leftover compositions into the training set
    n_train_comp = max(n_train_comp, nunique_compositions - n_valid_comp)
    assert n_train_comp + n_valid_comp == nunique_compositions

    # Shuffle the list for randomisation
    random.shuffle(unique_compositions)

    # Subset the list
    train_comps = unique_compositions[:n_train_comp]
    valid_comps = unique_compositions[n_train_comp:]

    # Sanity checks: should have desired number of compositions, and no composition should be shared between splits
    assert len(train_comps) + len(valid_comps) == nunique_compositions
    check_all_splits_unique(train_comps, valid_comps)

    # Subset the dataframe to get the name of the actual performances
    train_perfs = metadata_df[metadata_df["_composition_id"].isin(train_comps)]["fname"].tolist()
    # test_perfs = metadata_df[metadata_df["_composition_id"].isin(test_comps)]["fname"].tolist()
    valid_perfs = metadata_df[metadata_df["_composition_id"].isin(valid_comps)]["fname"].tolist()

    # Sanity check: should have used all performances, and all should be unique
    assert len(train_perfs) + len(valid_perfs) == len(metadata_df)
    check_all_splits_unique(train_perfs, valid_perfs)

    # Dump the data splits
    for name, df in zip(["train", "validation"], [train_perfs, valid_perfs]):
        split_fpath = os.path.join(PRETRAIN_SPLIT_DIR, f'{name}_pretraining_split.txt')
        with open(split_fpath, 'w') as f:
            for line in df:
                f.write(f"atepp/{line}\n")


def main():
    # Download the zip file
    out_path = download_file_from_google_drive()

    # Unzip it into the target directory
    subprocess.call(f"unzip {out_path} -d {DESTINATION}", shell=True)

    # Format the tracks: one folder per metadata/midi file
    all_metadatas = format_tracks()

    # Remove everything that is left over
    os.remove(out_path)
    shutil.rmtree(os.path.join(DESTINATION, "__MACOSX"))
    shutil.rmtree(os.path.join(DESTINATION, "ATEPP-1.2"))
    os.remove(os.path.join(DESTINATION, "ATEPP-metadata-1.2.csv"))

    # Create the data splits based on composition
    create_data_splits(all_metadatas)


if __name__ == "__main__":
    utils.seed_everything(utils.SEED)
    main()
