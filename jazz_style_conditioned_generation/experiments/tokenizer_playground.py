#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Experiment with different tokenizer types and vocab sizes, training for a few epochs with each and measuring loss"""

import os
from copy import deepcopy
from itertools import product
from time import time

import pandas as pd
from loguru import logger

from jazz_style_conditioned_generation import utils
from jazz_style_conditioned_generation.training import TrainingModule

TOKENIZER_TYPES = ["structured", "pertok"]
VOCAB_SIZES = [500, 750, 1000, 2500, 5000, 7500, 10000, 20000]
TEST_CONFIGS = list(product(TOKENIZER_TYPES, VOCAB_SIZES))

N_EPOCHS_PER_CONFIG = 3

TRAINING_CONFIG = {
    "experiment": "tokenizer-playground",
    "conditions": [],
    "batch_size": 2,
    "epochs": N_EPOCHS_PER_CONFIG,
    "train_dataset_cfg": {
        "do_augmentation": True,
        "do_conditioning": False
    },
    "test_dataset_cfg": {
        "do_augmentation": False,
        "do_conditioning": False
    },
    "model_cfg": {
        "model_type": "music-transformer",
        "model_kws": {
            "rpr": True
        }
    },
    "optimizer_cfg": {
        "optimizer_type": "adam",
        "optimizer_kws": {
            "lr": 2e-5
        }
    },
    "scheduler_cfg": {
        "scheduler_type": None,
        "scheduler_kws": {}
    },
    "checkpoint_cfg": {
        "save_checkpoints": False,
        "load_checkpoints": False
    },
    "generate_cfg": {
        "do_generation": False
    },
    "mlflow_cfg": {
        "use": False
    }
}


def init_training_module(tokenizer_type: str, vocab_size: int) -> TrainingModule:
    cfg = deepcopy(TRAINING_CONFIG)
    cfg["run"] = f"tokenizer_playground_{tokenizer_type}_{vocab_size}"
    cfg["tokenizer_cfg"] = {
        "tokenizer_str": tokenizer_type,
        "do_training": True,
        "training_method": "BPE",
        "vocab_size": vocab_size,
        "tokenizer_kws": {},
    }
    if tokenizer_type == "pertok":
        cfg["tokenizer_cfg"]["tokenizer_kws"] = {
            "use_microtiming": True,
            "ticks_per_quarter": 384,
            "max_microtiming_shift": 0.125,
            "num_microtiming_bins": 30,
        }
    return TrainingModule(**cfg)


def main():
    res = []
    logger.info("----------VOCAB SIZE EXPERIMENT----------")
    # Log the total number of experiments
    total = len(TEST_CONFIGS)
    logger.info(f"----------RUNNING {total} EXPERIMENTS----------")
    # Iterate through all the experiments we want to do
    for num, (tok_type, vocab_size) in enumerate(TEST_CONFIGS, 1):
        logger.info(f"----------EXPERIMENT {num}/{total}: TOK_TYPE {tok_type}, VOCAB_SIZE {vocab_size}------------")
        # Grab the training module
        tm = init_training_module(tok_type, vocab_size)
        # Iterate over all the epochs
        for epoch in range(tm.current_epoch, N_EPOCHS_PER_CONFIG):
            start_time = time()
            tm.current_epoch = epoch
            # Training
            train_loss, train_accuracy = tm.training(epoch)
            logger.debug(f'Epoch {epoch} / {tm.epochs}, training finished: '
                         f'loss {train_loss:.3f}, accuracy {train_accuracy:.3f}')
            # Validation
            validation_loss, validation_accuracy = tm.validation(epoch)
            logger.debug(f'Epoch {epoch} / {tm.epochs}, validation finished: '
                         f'loss {validation_loss:.3f}, accuracy {validation_accuracy:.3f}')
            # Append everything to the results list
            res.append({
                "tok_type": tok_type,
                "vocab_size": vocab_size,
                "epoch": epoch,
                "epoch_time": time() - start_time,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy
            })
    df = pd.DataFrame(res)
    df.to_csv(os.path.join(utils.get_project_root(), "references/vocab_size_experiment.csv"))


if __name__ == "__main__":
    # For reproducible results!
    utils.seed_everything(utils.SEED)
    main()
