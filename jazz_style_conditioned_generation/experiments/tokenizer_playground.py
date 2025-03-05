#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Experiment with different tokenizer types and vocab sizes, training for a few epochs with each and measuring loss"""

import os
from argparse import ArgumentParser
from copy import deepcopy
from itertools import product
from time import time

import pandas as pd
from loguru import logger

from jazz_style_conditioned_generation import utils
from jazz_style_conditioned_generation.training import TrainingModule

TOKENIZER_TYPES = ["structured", "pertok"]
VOCAB_SIZES = [500, 750, 1000, 2500, 5000, 7500, 10000, 20000]
TEST_CONFIGS = sorted(list(product(TOKENIZER_TYPES, VOCAB_SIZES)), key=lambda x: x[1])

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
    """Initialises the training module with the given tokenizer values"""
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


def main(tokenizers: list[str], vocabs: list[str], train_model: bool):
    logger.info("----------TOKENIZER PLAYGROUND----------")
    # Get the product of all tokenizers and vocab types, sort by vocab type (increasing)
    test_configs = sorted(list(product(tokenizers, vocabs)), key=lambda x: x[1])
    # Log the total number of experiments
    total = len(test_configs)
    logger.info(f"----------RUNNING {total} EXPERIMENTS----------")
    logger.info(f'EXPERIMENT CONFIGS: {test_configs}')
    # We'll use this to store the results if we're training the model
    res = []
    # Iterate through all the experiments we want to do
    for num, (tok_type, vocab_size) in enumerate(test_configs, 1):
        logger.info(f"----------EXPERIMENT {num}/{total}: TOK_TYPE {tok_type}, VOCAB_SIZE {vocab_size}------------")
        # Grab the training module, this will also train the tokenizer for us
        tm = init_training_module(tok_type, vocab_size)
        # If we don't want to train the model, we can just skip to the next tokenizer type + vocab size
        if not train_model:
            continue
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
    # If we've got results from training the models with each tokenizer, dump this to a CSV
    if len(res) > 1:
        df = pd.DataFrame(res)
        df.to_csv(os.path.join(utils.get_project_root(), "references/vocab_size_experiment.csv"))
    logger.info('Done!')


if __name__ == "__main__":
    # For reproducible results!
    utils.seed_everything(utils.SEED)
    # Parsing arguments from the command line interface
    parser = ArgumentParser(description="Experiment with training different tokenizer types + vocab sizes")
    parser.add_argument(
        '-t', '--tokenizers',
        nargs='+',
        help='Tokenizer types to use',
        default=TOKENIZER_TYPES
    )
    parser.add_argument(
        '-v', '--vocab-size',
        nargs='+',
        help='Vocab sizes to use',
        default=VOCAB_SIZES,
    )
    parser.add_argument(
        '-m', '--train-model',
        type=utils.string_to_bool,
        nargs='?',
        const=True,
        default=False,
        help="Whether or not to train the model as well as the tokenizer"
    )
    # Parse all arguments from the CLI
    args = vars(parser.parse_args())
    # Run the experiments with the given values
    main(args["tokenizers"], args["vocab_size"], args["train_model"])
