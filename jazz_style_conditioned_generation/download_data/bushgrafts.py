#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Downloads and preprocesses MIDI files from https://bushgrafts.com/midi/"""

import json
import os
import shutil
from secrets import token_hex

import requests
from loguru import logger
from lxml import html
from pretty_midi import PrettyMIDI
from tqdm import tqdm

from jazz_style_conditioned_generation import utils

WEB_ROOT = "https://bushgrafts.com/jazz/Midi%20site/"
OUTPUT_DIR = os.path.join(utils.get_project_root(), "data/raw/bushgrafts")


def download_file(url: str) -> str:
    local_filename = url.split('/')[-1]
    with requests.get(url, stream=True) as r:
        with open(local_filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    return local_filename


def scrape_urls(root: str = WEB_ROOT) -> list[str]:
    page = requests.get(root)
    webpage = html.fromstring(page.content)
    hrefs = webpage.xpath('//a/@href')
    return [os.path.join(root, i) for i in hrefs if i.lower().endswith('.mid')]


def validate_midi(midi_str: str) -> tuple[PrettyMIDI, str] | None:
    try:
        pm = PrettyMIDI(midi_str)
    except ValueError as e:
        logger.error(f'File {midi_str} raised error {e} when converting to MIDI, skipping!')
        return None
    # Remove any text events (i.e., Doug's annotations) from the MIDI file
    pm.text_events = []
    pm.key_signature_changes = []
    pm.time_signature_changes = []
    # We just have one instrument, so it'll be piano
    if len(pm.instruments) == 1:
        if pm.instruments[0].program == utils.MIDI_PIANO_PROGRAM:
            return pm, "unaccompanied"
        else:
            logger.error(f'File {midi_str} contains one instrument but the '
                         f'program is not {utils.MIDI_PIANO_PROGRAM}, skipping!')
            return None
    else:
        try:
            # Get the instrument which has the correct program
            piano = [i for i in pm.instruments if i.program == utils.MIDI_PIANO_PROGRAM][0]
        except IndexError:
            logger.error(f'File {midi_str} does not have any instruments '
                         f'with program == {utils.MIDI_PIANO_PROGRAM}, skipping!')
            return None
        else:
            # Subset the instruments to remove all non-piano instruments
            newmid = PrettyMIDI()
            newmid.instruments = [piano]
            return newmid, "bassbdrumsd"


def format_trackname(fp: str) -> str:
    return utils.remove_punctuation(
        fp.lower()
        .replace("%20", " ")
        .replace(".mid", "")
        .split(" ")[:6]
    ).replace(" ", "")


def main():
    urls = scrape_urls()
    logger.info(f"Found {len(urls)} URLs to process")
    for url in tqdm(urls, desc="Processing MIDIs..."):
        tmp_fp = download_file(url)
        # Will return a tuple of (PrettyMIDI, ensemble type) if valid or None if invalid
        validated = validate_midi(tmp_fp)
        if validated is not None:
            midi, context = validated
            track_name = format_trackname(tmp_fp)
            # We use a unified format of artist-track-ensemble-year-mbzid across all tracks
            # We don't have years or musicbrainz ids for these recordings, so we can just use a placeholder
            hasher = token_hex(4)
            new_fp = f'mckenzied-{track_name}-{context}-xxxx-{hasher}'
            # Create the directory we're saving everything to do with this track in
            full_path = os.path.join(OUTPUT_DIR, new_fp)
            if not os.path.isdir(full_path):
                os.makedirs(full_path)
            # Save the MIDI
            midi.write(os.path.join(full_path, "piano_midi.mid"))
            # Create the metadata
            metadata = {
                "track_name": track_name,
                "album_name": "Bushgrafts",
                "recording_year": 2025,
                "bandleader": "Doug McKenzie",
                "pianist": "Doug McKenzie",
                # Scramble and create a random "musicbrainz-like" ID
                "mbz_id": f"{hasher}-{token_hex(2)}-{token_hex(2)}-{token_hex(2)}-{token_hex(6)}",
                "live": False,
                "fname": new_fp
            }
            # Save the metadata
            with open(os.path.join(full_path, "metadata.json"), "w") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=4)
        # Tidy up by removing the temporary MIDI file we created
        os.remove(tmp_fp)
    logger.info("Done")


if __name__ == '__main__':
    utils.seed_everything()
    main()
