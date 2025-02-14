import requests
import os
import json

from jazz_style_conditioned_generation import utils

DATA_ROOT = os.path.join(utils.get_project_root(), 'data/raw')
DATASETS = ['jtd', 'pijama']
print(DATA_ROOT)

API_ROOT = "https://tivomusicapi-staging-elb.digitalsmiths.net/sd/tivomusicapi/taps/v3"
API_ALBUM_SEARCH = f'{API_ROOT}/search/album'
API_ALBUM_LOOKUP = f'{API_ROOT}/lookup/album'


def get_tracks():
    for dataset in DATASETS:
        dataset_dir = os.path.join(DATA_ROOT, dataset)
        for track in os.listdir(dataset_dir):
            track_dir = os.path.join(dataset_dir, track)
            if not os.path.isdir(track_dir):
                continue
            metadata_path = os.path.join(track_dir, 'metadata.json')
            if not os.path.isfile(metadata_path):
                raise FileNotFoundError(f'Could not find metadata for track {track}, dataset {dataset}')
            yield track_dir


def read_track_metadata(track_dir):
    with open(os.path.join(track_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    return metadata


def make_api_request(track_metadata):
    pass


def validate_api_hit(track_metadata: dict, track_hit: dict):
    pass


def main():
    all_tracks = get_tracks()
    for track in all_tracks:
        track_metadata = read_track_metadata(track)
        pass


if __name__ == "__main__":
    print(DATA_ROOT)
    assert os.path.isdir(DATA_ROOT), f'Could not find data at {DATA_ROOT}!'
    main()
