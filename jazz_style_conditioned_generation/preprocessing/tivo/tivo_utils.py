#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility functions, constants etc. used throughout scraping TiVo API"""

import os
import re
from functools import lru_cache

import requests

from jazz_style_conditioned_generation import utils

DATA_ROOT = os.path.join(utils.get_project_root(), 'data/raw')
TIVO_DATASETS = ["jtd", 'pijama', 'pianist8', ]  # datasets with no metadata: jja, bushgrafts

API_ROOT = "https://tivomusicapi-staging-elb.digitalsmiths.net/sd/tivomusicapi/taps/v3"
API_HEADERS = {"Accept": "application/json"}
API_WAIT_TIME = 0.1  # seconds, API terms of use specify no more than five calls per second


def add_missing_keys(di: dict, keys: list, value_type: type = list) -> dict:
    """Adds key-value pairs that do not exist into a dictionary"""
    for key in keys:
        if key not in di.keys():
            di[key] = value_type()  # defaults to an empty list
    return di


def format_named_person_or_entity(npe: str):
    return " ".join(npe.lstrip().rstrip().title().split())


@utils.wait(secs=API_WAIT_TIME)
@lru_cache(maxsize=None)
def cached_api_call(url: str) -> dict:
    """Makes an API call to a given url: waiting and caching are implemented to prevent rate limiting"""
    return requests.get(url, headers=API_HEADERS).json()


def clean_prose_text(prose_text: str) -> str:
    """Prose text from TiVo (e.g., bios, reviews) contains some HTML tags which we need to remove"""
    # Iterate over each markup tag
    for remove in ["roviLink", "muzeItalic"]:
        # Use some regex to remove the tags but keep the content between the tags
        prose_text = re.sub(rf'\[{remove}.*?\](.*?)\[/{remove}\]', r'\1', prose_text)
    # This character is typically used to indicate the name of the reviewer at the end of the review
    if "~" in prose_text[-50:]:
        prose_text = prose_text[:-50] + prose_text[-50:].split("~")[0]
    return prose_text.rstrip()
