#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Downloads and preprocesses MIDI files from https://zenodo.org/records/5089279"""

import json
import os
import shutil
import subprocess
from secrets import token_hex

from jazz_style_conditioned_generation import utils

WEB_ROOT = "https://zenodo.org/records/5089279/files/joann8512/Pianist8-v1.0.0.zip?download=1"
# These files are also contained in PiJAMA: we keep the PiJAMA versions
MAPPINGS = {
    "Hancock_All_Apologies.mid": {
        "album_name": "The New Standard",
        "recording_year": 1996,
    },
    "Hancock_Alone_and_I.mid": {
        "album_name": "Takin' Off",
        "recording_year": 1962,
    },
    "Hancock_Amelia.mid": {
        "album_name": "River The Joni Letters",
        "recording_year": 2007,
    },
    "Hancock_And_What_If_I_Don't.mid": {
        "track_name": "And What If I Don't Know",
        "album_name": "My Point Of View",
        "recording_year": 1999,
    },
    "Hancock_Blind_Man,_Blind_Man.mid": {
        "album_name": "My Point Of View",
        "recording_year": 1999,
    },
    "Hancock_Both_Sides_Now.mid": {
        "album_name": "River The Joni Letters",
        "recording_year": 2007,
    },
    "Hancock_Button_Up.mid": {
        "album_name": "An Evening With Herbie Hancock & Chick Corea In Concert",
        "recording_year": 1978,
        "live": True,
    },
    "Hancock_Court_and_Spark.mid": {
        "album_name": "River The Joni Letters",
        "recording_year": 2007,
    },
    "Hancock_Dolphin_Dance.mid": {
        "album_name": "Maiden Voyage",
        "recording_year": 1965,
    },
    "Hancock_Driftin.mid": {
        "album_name": "Takin' Off",
        "recording_year": 1962,
    },
    "Hancock_Edith_and_the_Kingpin.mid": {
        "album_name": "River The Joni Letters",
        "recording_year": 2007,
    },
    "Hancock_Empty_Pockets.mid": {
        "album_name": "Takin' Off",
        "recording_year": 1962,
    },
    "Hancock_February_Moment.mid": {
        "album_name": "An Evening With Herbie Hancock & Chick Corea In Concert",
        "recording_year": 1978,
        "live": True,
    },
    "Hancock_First_Trip.mid": {
        "album_name": "Speak Like a Child",
        "recording_year": 1968,
    },
    "Hancock_It_Ain't_Necessarily_So.mid": {
        "album_name": "Gershwin's World",
        "recording_year": 1998,
    },
    "Hancock_Jessica.mid": {
        "album_name": "Fat Albert Rotunda",
        "recording_year": 1969,
    },
    "Hancock_La_Fiesta.mid": {
        "album_name": "An Evening With Herbie Hancock & Chick Corea In Concert",
        "recording_year": 1978,
        "live": True,
    },
    "Hancock_Little_One.mid": {
        "album_name": "Maiden Voyage",
        "recording_year": 1965,
    },
    "Hancock_Liza.mid": {
        "album_name": "An Evening With Herbie Hancock & Chick Corea In Concert",
        "recording_year": 1978,
        "live": True,
    },
    "Hancock_Love_is_Stronger_than_Pride.mid": {
        "album_name": "The New Standard",
        "recording_year": 1996,
    },
    "Hancock_Maiden_Voyage.mid": {
        "album_name": "Maiden Voyage",
        "recording_year": 1965,
    },
    "Hancock_Manhattan.mid": {
        "album_name": "The New Standard",
        "recording_year": 1996,
    },
    "Hancock_Nefertiti.mid": {
        "album_name": "River The Joni Letters",
        "recording_year": 2007,
    },
    "Hancock_New_York_Minute.mid": {
        "album_name": "The New Standard",
        "recording_year": 1996,
    },
    "Hancock_Norwegian_Wood.mid": {
        "album_name": "The New Standard",
        "recording_year": 1996,
    },
    "Hancock_River.mid": {
        "album_name": "River The Joni Letters",
        "recording_year": 2007,
    },
    "Hancock_Scarborough_Fair.mid": {
        "album_name": "The New Standard",
        "recording_year": 1996,
    },
    "Hancock_Solitude.mid": {
        "album_name": "River The Joni Letters",
        "recording_year": 2007,
    },
    "Hancock_Speak_Like_a_Child.mid": {
        "album_name": "Speak Like a Child",
        "recording_year": 1968,
    },
    "Hancock_Survival_of_the_Fittest.mid": {
        "album_name": "Maiden Voyage",
        "recording_year": 1965,
    },
    "Hancock_Sweet_Bird.mid": {
        "album_name": "River The Joni Letters",
        "recording_year": 2007,
    },
    "Hancock_The_Eye_of_the_Hurricane.mid": {
        "album_name": "Maiden Voyage",
        "recording_year": 1965,
    },
    "Hancock_The_Jungle_Line.mid": {
        "album_name": "River The Joni Letters",
        "recording_year": 2007,
    },
    "Hancock_The_Man_I_Love.mid": {
        "album_name": "Gershwin's World",
        "recording_year": 1998,
    },
    "Hancock_The_Maze.mid": {
        "album_name": "Takin' Off",
        "recording_year": 1962,
    },
    "Hancock_The_Pleasure_Is_Mine.mid": {
        "album_name": "My Point Of View",
        "recording_year": 1999,
    },
    "Hancock_The_Sorcerer.mid": {
        "album_name": "Speak Like a Child",
        "recording_year": 1968,
    },
    "Hancock_The_Tea_Leaf_Prophecy.mid": {
        "album_name": "River The Joni Letters",
        "recording_year": 2007,
    },
    "Hancock_Three_Bags_Full.mid": {
        "album_name": "Takin' Off",
        "recording_year": 1962,
    },
    "Hancock_Thieves_in_the_Temple.mid": {
        "album_name": "The New Standard",
        "recording_year": 1996,
    },
    "Hancock_Toys.mid": {
        "album_name": "Speak Like a Child",
        "recording_year": 1968,
    },
    "Hancock_Unknown1.mid": {
        "album_name": "Unknown",
        "recording_year": 2000,
    },
    "Hancock_Unknown2.mid": {
        "album_name": "Unknown",
        "recording_year": 2000,
    },
    "Hancock_Watermelon_Man.mid": {
        "album_name": "Takin' Off",
        "recording_year": 1962,
    },
    "Hancock_When_I_Can_See_You.mid": {
        "album_name": "The New Standard",
        "recording_year": 1996,
    },
    "Hancock_Your_Gold_Teeth.mid": {
        "track_name": "Your Gold Teeth II",
        "album_name": "The New Standard",
        "recording_year": 1996,
    },
    "Hancock_Youve_Got_It_Bad_Girl.mid": {
        "track_name": "You've Got It Bad Girl",
        "album_name": "The New Standard",
        "recording_year": 1996,
    },
}


def zenodo_download_and_unzip():
    subprocess.call(f"wget -r {WEB_ROOT} -O pianist8.zip", shell=True)
    subprocess.call("unzip pianist8.zip", shell=True)


def process_midis():
    root = "joann8512-Pianist8-ab9f541/midi/Hancock"
    for f in os.listdir(root):
        if f not in MAPPINGS.keys():
            continue
        metadata = MAPPINGS[f]
        if "track_name" not in metadata.keys():
            track_name = f.replace("Hancock_", "").replace(".mid", "").replace("_", " ")
            metadata["track_name"] = track_name
        else:
            track_name = metadata["track_name"]

        if "live" not in metadata.keys():
            metadata["live"] = False

        metadata["bandleader"] = "Herbie Hancock"
        metadata["pianist"] = "Herbie Hancock"
        track_name_fmt = "".join(utils.remove_punctuation(track_name).lower().split(" ")[:6])
        year = metadata["recording_year"]
        hexe = token_hex(4)
        fpath = f"hancockh-{track_name_fmt}-bassbdrumsd-{year}-{hexe}"
        metadata["fname"] = fpath
        metadata["mbz_id"] = f"{hexe}-{token_hex(2)}-{token_hex(2)}-{token_hex(2)}-{token_hex(6)}"

        src = os.path.join(root, f)
        dest = os.path.join(utils.get_project_root(), "data/raw/pianist8", fpath)
        if not os.path.isdir(dest):
            os.makedirs(dest)

        dest_fp = os.path.join(dest, "piano_midi.mid")
        os.rename(src, dest_fp)

        with open(os.path.join(dest, "metadata.json"), "w") as outp:
            json.dump(metadata, outp, ensure_ascii=True, indent=4)


def main():
    zenodo_download_and_unzip()
    process_midis()
    shutil.rmtree("joann8512-Pianist8-ab9f541")
    os.remove("pianist8.zip")


if __name__ == "__main__":
    main()
