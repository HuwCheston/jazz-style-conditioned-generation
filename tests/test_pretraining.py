#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for pre-training module, won't run on GitHub actions"""

import os
import unittest
from copy import deepcopy

import torch

from jazz_style_conditioned_generation import utils, training
from jazz_style_conditioned_generation.pretraining import PreTrainingModule


def handle_cuda_exceptions(f):
    """Skips a test when we get a CUDA out-of-memory error, allowing tests to run parallel with training runs."""

    def wrapper(*args, **kw):
        try:
            return f(*args, **kw)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            unittest.skip("Ignoring CUDA out of memory error!")

    return wrapper


# @handle_cuda_exceptions
@unittest.skipIf(os.getenv("REMOTE") == "true", "Skipping test on GitHub Actions")
class PreTrainTests(unittest.TestCase):
    def setUp(self):
        yaml_path = os.path.join(utils.get_project_root(), "tests/test_resources/train_config.yaml")
        self.CONFIG = training.parse_config_yaml(yaml_path)
        self.CONFIG["tokenizer_cfg"]["do_training"] = False  # skip training the tokenizer, for simplicity
        self.PRETRAINER = PreTrainingModule(**self.CONFIG)
        self.CONDITION_TOKENS = ("PIANIST", "GENRES", "RECORDINGYEAR", "TEMPO", "TIMESIGNATURE")

    @handle_cuda_exceptions
    def test_with_conditioning(self):
        cfg_copy = deepcopy(self.CONFIG)
        # Setting conditioning to True: we should HAVE condition tokens in our tokenizer/model FC layer
        #  This is to ensure compatability when we come to fine-tune, and we want the model to have enough
        #  channels in the fully-connected layer.
        #  However, we should internally set the do_conditioning argument to False in our dataloader
        #  This is so we don't try and get condition tokens for our pretraining data
        cfg_copy["test_dataset_cfg"]["do_conditioning"] = True
        cfg_copy["train_dataset_cfg"]["do_conditioning"] = True
        tm = PreTrainingModule(**cfg_copy)
        # Check arguments propagated correctly
        self.assertTrue(tm.train_dataset_cfg["do_conditioning"])
        self.assertTrue(tm.test_dataset_cfg["do_conditioning"])
        # Check that we have our condition tokens in the vocabulary
        for condition_token_start in self.CONDITION_TOKENS:
            condition_tokens = [i for i in tm.tokenizer.vocab.keys() if i.startswith(condition_token_start)]
            self.assertTrue(len(condition_tokens) > 0)
        # Check that our final FC layer has the correct number of outputs
        self.assertTrue(tm.model.Wout.out_features == tm.tokenizer.vocab_size)
        # Internally within our dataloaders, do_conditioning should be set to false
        self.assertFalse(tm.train_loader.dataset.do_conditioning)
        self.assertFalse(tm.test_loader.dataset.do_conditioning)
        self.assertFalse(tm.validation_loader.dataset.do_conditioning)
        # Calling __getitem__ on the dataloaders, we shouldn't have any condition tokens
        for dataloader in [tm.train_loader, tm.test_loader, tm.validation_loader]:
            for batch_idx in range(10):
                batch = dataloader.dataset.__getitem__(batch_idx)
                iids = batch["input_ids"].tolist()
                decoded = [tm.tokenizer[i] for i in iids]
                for tok in decoded:
                    self.assertFalse(tok.startswith(self.CONDITION_TOKENS))

    @handle_cuda_exceptions
    def test_without_conditioning(self):
        # Check arguments propagated correctly
        self.assertFalse(self.PRETRAINER.train_dataset_cfg["do_conditioning"])
        self.assertFalse(self.PRETRAINER.test_dataset_cfg["do_conditioning"])
        # Check that we do not have our condition tokens in the vocabulary
        for condition_token_start in self.CONDITION_TOKENS:
            condition_tokens = [i for i in self.PRETRAINER.tokenizer.vocab.keys() if
                                i.startswith(condition_token_start)]
            self.assertFalse(len(condition_tokens) > 0)
        # Check that our final FC layer has the correct number of outputs
        self.assertTrue(self.PRETRAINER.model.Wout.out_features == self.PRETRAINER.tokenizer.vocab_size)
        # Internally within our dataloaders, do_conditioning should be set to false
        self.assertFalse(self.PRETRAINER.train_loader.dataset.do_conditioning)
        self.assertFalse(self.PRETRAINER.test_loader.dataset.do_conditioning)
        self.assertFalse(self.PRETRAINER.validation_loader.dataset.do_conditioning)

    def test_dataset_items(self):
        for item in self.PRETRAINER.track_paths:
            self.assertTrue("atepp" in item)


if __name__ == '__main__':
    utils.seed_everything(utils.SEED)
    unittest.main()
