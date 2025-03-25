#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Pretrain a model on ATEPP before training on the jazz dataset. Overrides training.py modules"""

import os
from copy import deepcopy

from loguru import logger
from torch.utils.data import DataLoader

from jazz_style_conditioned_generation import utils, training
from jazz_style_conditioned_generation.data.dataloader import (
    DatasetMIDIConditionedRandomChunk,
    DatasetMIDIConditionedFullTrack
)


class PreTrainingModule(training.TrainingModule):
    def __init__(self, *args, **kwargs):
        logger.info("----PRETRAINING ON ATEPP----")
        super().__init__(*args, **kwargs)

    def create_dataloaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Overrides base methods to force creating dataloaders without conditioning"""
        # Create validation dataset loader: uses random chunks
        if self.test_dataset_cfg.get("do_augmentation", False):
            raise AttributeError("Augmentation only allowed for training dataloader!")

        # Copy the configuration dictionary and remove the `do_conditioning` argument
        test_kws = deepcopy(self.test_dataset_cfg)
        test_kws.pop("do_conditioning", None)
        # Create test dataset loader: uses FULL tracks!
        test_loader = DataLoader(
            DatasetMIDIConditionedFullTrack(
                tokenizer=self.tokenizer,
                files_paths=self.track_splits["test"],
                max_seq_len=utils.MAX_SEQUENCE_LENGTH,
                do_conditioning=False,
                **test_kws  # most arguments can be shared across test + validation loader
            ),
            batch_size=1,  # have to use a batch size of one for this class
            shuffle=False,  # don't want to shuffle either for this one
            drop_last=False,
        )
        if self._generate_only:
            return None, None, test_loader  # hack to avoid creating other dataloaders when we don't want them

        validation_loader = DataLoader(
            DatasetMIDIConditionedRandomChunk(
                tokenizer=self.tokenizer,
                files_paths=self.track_splits["validation"],
                max_seq_len=utils.MAX_SEQUENCE_LENGTH,
                do_conditioning=False,
                **test_kws  # most arguments can be shared across test + validation loader
            ),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )
        # Copy the configuration dictionary and remove the `do_conditioning` argument
        train_kws = deepcopy(self.train_dataset_cfg)
        train_kws.pop("do_conditioning", None)
        # Create test dataset loader: uses random chunks
        train_loader = DataLoader(
            DatasetMIDIConditionedRandomChunk(
                tokenizer=self.tokenizer,
                files_paths=self.track_splits["train"],
                max_seq_len=utils.MAX_SEQUENCE_LENGTH,
                do_conditioning=False,
                **train_kws
            ),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        return train_loader, validation_loader, test_loader

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
    # training_kwargs["test_dataset_cfg"]["do_conditioning"] = False  # should never condition during pretraining
    # training_kwargs["train_dataset_cfg"]["do_conditioning"] = False
    # training_kwargs["scheduler_cfg"]["scheduler_type"] = None    # should not use an LR scheduler during pretraining
    training_kwargs["scheduler_cfg"]["do_early_stopping"] = False  # no early stopping during pretraining
    # Start training!
    training.main(training_kwargs, trainer_cls=PreTrainingModule, config_fpath=parser_args["config"])
