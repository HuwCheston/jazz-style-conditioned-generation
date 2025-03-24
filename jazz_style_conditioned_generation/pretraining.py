#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Pretrain a model on ATEPP before training on the jazz dataset. Overrides training.py modules"""

import os

from loguru import logger

from jazz_style_conditioned_generation import utils, training


class PreTrainingModule(training.TrainingModule):
    def __init__(self, *args, **kwargs):
        logger.info("----PRETRAINING ON ATEPP----")
        super().__init__(*args, **kwargs)

    def read_tracks_for_split(self, split_type: str) -> list[str]:
        """Reads a txt file containing a one line per string and returns as a list of strings"""
        split_fp = os.path.join(self.split_dir, split_type + '_pretraining_split.txt')
        with open(split_fp, 'r') as fp:
            all_paths = fp.read().strip().split('\n')
            # Check that the path exists on the local file structure
            for path in all_paths:
                track_path = os.path.join(self.data_dir, path, "piano_midi.mid")
                if not os.path.isfile(track_path):
                    raise FileNotFoundError(f'Could not find MIDI for track at {track_path}')
                # No need for metadata for the pretraining dataset
                yield track_path

    def save_checkpoint(self, epoch_metrics: dict, path: str) -> None:
        epoch_metrics["pretraining"] = True  # add a flag to the checkpoint
        # path = path.replace(".pth", "_pretraining.pth")     # add to the filename
        super().save_checkpoint(epoch_metrics, path)  # save the checkpoint as normal

    @property
    def data_dir(self) -> str:
        return os.path.join(utils.get_project_root(), "data/pretraining")

    @property
    def split_dir(self) -> str:
        return os.path.join(utils.get_project_root(), "references/data_splits/pretraining")


if __name__ == '__main__':
    import argparse

    # Seed everything for reproducible results
    utils.seed_everything(utils.SEED)

    # Parsing arguments from the command line interface
    parser = argparse.ArgumentParser(description="Run model training")
    parser.add_argument("-c", "--config", default=None, type=str, help="Path to config YAML file")
    # Parse all arguments from the provided YAML file
    parser_args = vars(parser.parse_args())
    if not parser_args:
        raise ValueError("No config file specified")
    training_kwargs = training.parse_config_yaml(parser_args['config'])
    # Hardcode a few assumptions for pretraining
    training_kwargs["_generate_only"] = False  # should only be set to True when running generate.py
    training_kwargs["test_dataset_cfg"]["do_conditioning"] = False  # should never condition during pretraining
    training_kwargs["train_dataset_cfg"]["do_conditioning"] = False
    # training_kwargs["scheduler_cfg"]["scheduler_type"] = None    # should not use an LR scheduler during pretraining
    training_kwargs["scheduler_cfg"]["do_early_stopping"] = False  # no early stopping during pretraining
    # Start training!
    training.main(training_kwargs, trainer_cls=PreTrainingModule, config_fpath=parser_args["config"])
