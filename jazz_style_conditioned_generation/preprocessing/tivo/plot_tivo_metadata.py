import json
import os

import pandas as pd

from jazz_style_conditioned_generation import utils, plotting

GENRES_TO_DROP = ['Jazz', 'Piano Jazz', 'Jazz Instrument']
DATA_ROOT = os.path.join(utils.get_project_root(), 'data/raw')
OUTPUT_ROOT = os.path.join(utils.get_project_root(), 'outputs/figures/tivo_metadata')
DATASETS = ["jtd", 'pijama']

N_PIANISTS = 10
N_UNIQUE_TAGS = 30
TAGS_TO_PLOT = ["genres", "moods", "themes"]


def get_metadata() -> pd.DataFrame:
    res = []
    for dataset in DATASETS:
        dataset_root = os.path.join(DATA_ROOT, dataset)
        for track in os.listdir(dataset_root):
            track_root = os.path.join(dataset_root, track)
            if not os.path.isdir(track_root):
                continue
            metadata_path = os.path.join(track_root, 'metadata_tivo.json')
            assert os.path.isfile(metadata_path), "Could not find metadata at {}!".format(metadata_path)
            metadata = json.load(open(metadata_path, 'r'))
            for key in TAGS_TO_PLOT:
                for value in [i["name"] for i in metadata[f'album_{key}']]:
                    if value in GENRES_TO_DROP and key == 'genres':
                        continue
                    res.append(dict(
                        id=metadata['mbz_id'],
                        pianist=metadata['pianist'],
                        track=metadata['track_name'],
                        album=metadata['album_name'],
                        unaccompanied=dataset == "pijama",
                        key=key,
                        value=value
                    ))
    return pd.DataFrame(res)


def get_most_common_pianists(df: pd.DataFrame, n_pianists: int = N_PIANISTS):
    return (
        df.drop_duplicates(subset=['id'])
        ['pianist']
        .value_counts()
        .sort_values(ascending=False)
        .head(n_pianists)
        .index
        .tolist()
    )


def truncate_df(df: pd.DataFrame, most_common_pianists: list[str], most_common_tags: list[str]):
    return (
        df[
            (df['pianist'].isin(most_common_pianists)) &
            (df['value'].isin(most_common_tags))
            ]
        .groupby(['pianist', 'key'])
        ['value']
        .value_counts()
        .reset_index(drop=False)
        .drop(columns='key')
    )


def get_most_common_tags(df: pd.DataFrame, tag_str: str = "themes", n_unique_tags: int = N_UNIQUE_TAGS):
    return (
        df[df['key'] == tag_str]
        ['value']
        .value_counts()
        .sort_values(ascending=False)
        .head(n_unique_tags)
        .index
        .tolist()
    )


def main():
    metadata = get_metadata()
    most_common_pianists = get_most_common_pianists(metadata)
    for tag in TAGS_TO_PLOT:
        most_common_tags = get_most_common_tags(metadata, tag, )
        truncated = truncate_df(metadata, most_common_pianists, most_common_tags)
        plotter = plotting.BarPlotTiVoMetadataTagCounts(truncated, tag_str=tag)
        plotter.create_plot()
        plotter.save_fig(os.path.join(OUTPUT_ROOT, 'barplot_tag_counts'))


if __name__ == "__main__":
    utils.seed_everything()
    main()
