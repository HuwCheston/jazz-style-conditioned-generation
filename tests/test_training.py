#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for training module"""

import os
import unittest
from copy import deepcopy

import torch

from jazz_style_conditioned_generation import utils
from jazz_style_conditioned_generation.encoders.music_transformer import MusicTransformer
from jazz_style_conditioned_generation.training import (
    TrainingModule,
    parse_config_yaml,
    DummyModule,
    PreTrainingModule,
)


def handle_cuda_exceptions(f):
    """Skips a test when we get a CUDA out-of-memory error, allowing tests to run parallel with training runs."""

    def wrapper(*args, **kw):
        try:
            return f(*args, **kw)
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            unittest.skip("Ignoring CUDA out of memory error!")

    return wrapper


class TrainingTest(unittest.TestCase):
    def setUp(self):
        yaml_path = os.path.join(utils.get_project_root(), "tests/test_resources/train_config.yaml")
        self.CONFIG = parse_config_yaml(yaml_path)
        try:
            self.TRAINER = TrainingModule(**self.CONFIG)
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            raise unittest.SkipTest("Ignoring CUDA out of memory error!")

    def test_dummy_module(self):
        dummy = DummyModule()
        self.assertEqual(utils.total_parameters(dummy), 1)
        # Forward function should just return input
        dummy_tensor = torch.randint(0, 100, (10,))
        thru = dummy(dummy_tensor)
        self.assertTrue(torch.equal(dummy_tensor, thru))

    def test_initialise_from_config(self):
        self.assertEqual(self.TRAINER.current_epoch, 0)  # not loading a checkpoint
        # Testing track and split paths
        self.assertEqual(len(self.TRAINER.track_paths), 3)
        for split in ["train", "test", "validation"]:
            self.assertEqual(len(self.TRAINER.track_splits[split]), 1)
        # Testing tokenizer
        self.assertFalse(self.TRAINER.tokenizer.is_trained)  # config is specifying no training
        # Testing training dataloader
        # self.assertEqual(len(self.TRAINER.train_loader.dataset), 1)  # uses random chunks
        self.assertFalse(self.TRAINER.train_loader.dataset.do_augmentation)  # no augmentation as specified in config
        self.assertEqual(len(self.TRAINER.test_loader.dataset), 1)  # random chunks
        self.assertEqual(len(self.TRAINER.validation_loader.dataset), 1)  # random chunks
        # Should have a maximum sequence length of 512 tokens per loader
        for loader in [self.TRAINER.train_loader, self.TRAINER.test_loader, self.TRAINER.validation_loader]:
            self.assertEqual(loader.dataset.max_seq_len, 512)
        # Testing model
        self.assertTrue(isinstance(self.TRAINER.model, MusicTransformer))
        self.assertTrue(self.TRAINER.model.rpr)  # config specifies to use RPR
        self.assertTrue(self.TRAINER.model.max_seq_len, 512)  # config specifies max sequence of 512

    def test_paths(self):
        # Passing these in from the config file
        expected_data_path = os.path.join(utils.get_project_root(), "tests/test_resources")
        self.assertEqual(expected_data_path, self.TRAINER.data_dir)
        expected_splits_path = os.path.join(utils.get_project_root(), "tests/test_resources/splits")
        self.assertEqual(expected_splits_path, self.TRAINER.split_dir)
        expected_checkpoints_path = os.path.join(utils.get_project_root(), "checkpoints/tester/tester_config")
        self.assertEqual(expected_checkpoints_path, self.TRAINER.checkpoint_dir)
        expected_generation_path = os.path.join(utils.get_project_root(), "outputs/generation/tester/tester_config")
        self.assertEqual(expected_generation_path, self.TRAINER.output_midi_dir)

    @handle_cuda_exceptions
    def test_step(self):
        batch = next(iter(self.TRAINER.train_loader))
        loss, accuracy = self.TRAINER.step(batch)
        self.assertTrue(loss.requires_grad)
        self.assertTrue(0. <= accuracy <= 1.)

    @handle_cuda_exceptions
    def test_train(self):
        # Get model parameters and make a copy for later comparison
        params = [np for np in self.TRAINER.model.named_parameters() if np[1].requires_grad]
        initial_params = [(name, p.clone()) for (name, p) in params]
        # Do a single training "epoch"
        train_loss, train_acc = self.TRAINER.training(0)
        self.assertTrue(isinstance(train_loss, float))
        self.assertTrue(isinstance(train_acc, float))
        self.assertTrue(0. <= train_acc <= 1.)
        # Model should be in training mode
        self.assertTrue(self.TRAINER.model.training)
        # Iterate through all the parameters in the model: they should have updated
        for (_, p0), (name, p1) in zip(initial_params, params):
            # If vars have changed, will return True; if not, will return False
            self.assertTrue(not torch.equal(p0.to(utils.DEVICE), p1.to(utils.DEVICE)))

    @handle_cuda_exceptions
    def test_eval(self):
        valid_loss, valid_acc = self.TRAINER.validation(0)
        self.assertTrue(isinstance(valid_loss, float))
        self.assertTrue(isinstance(valid_acc, float))
        self.assertTrue(0. <= valid_acc <= 1.)
        # test_loss, test_acc = self.TRAINER.testing()
        # self.assertTrue(isinstance(test_loss, float))
        # self.assertTrue(isinstance(test_acc, float))
        # self.assertTrue(0. <= test_acc <= 1.)
        # Model should be put into validation mode
        self.assertFalse(self.TRAINER.model.training)

    @handle_cuda_exceptions
    def test_checkpointing(self):
        # Save a dummy checkpoint
        temp_checkpoint_name = f"{self.TRAINER.checkpoint_dir}/temp_checkpoint_001.pth"
        temp_metrics = dict(hello="there")
        self.TRAINER.save_checkpoint(temp_metrics, temp_checkpoint_name)
        self.assertTrue(os.path.exists(temp_checkpoint_name))
        # Do some training
        # Get model parameters and make a copy for later comparison
        params = [np for np in self.TRAINER.model.named_parameters() if np[1].requires_grad]
        initial_params = [(name, p.clone()) for (name, p) in params]
        _, ___ = self.TRAINER.training(0)
        # Iterate through all the parameters in the model: they should have updated
        for (_, p0), (name, p1) in zip(initial_params, params):
            # If vars have changed, will return True; if not, will return False
            self.assertTrue(not torch.equal(p0.to(utils.DEVICE), p1.to(utils.DEVICE)))
        # RELOAD our initial checkpoint
        self.TRAINER.load_most_recent_checkpoint()
        # Parameters should be the same as the initial values
        for (_, p0), (name, p1) in zip(initial_params, params):
            # If vars have changed, will return True; if not, will return False
            self.assertTrue(torch.equal(p0.to(utils.DEVICE), p1.to(utils.DEVICE)))
        # Remove the checkpoint
        self.TRAINER.current_epoch = 100
        self.TRAINER.remove_old_checkpoints()
        self.assertFalse(os.path.exists(temp_checkpoint_name))
        # Reset everything back
        self.TRAINER.current_epoch = 0


class PreTrainTests(unittest.TestCase):
    def setUp(self):
        yaml_path = os.path.join(utils.get_project_root(), "tests/test_resources/pretrain_config.yaml")
        self.CONFIG = parse_config_yaml(yaml_path)
        self.CONFIG["tokenizer_cfg"]["do_training"] = False  # skip training the tokenizer, for simplicity
        self.CONDITION_TOKENS = ("PIANIST", "GENRES", "RECORDINGYEAR", "TEMPO", "TIMESIGNATURE")
        try:
            self.PRETRAINER = PreTrainingModule(**self.CONFIG)
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            raise unittest.SkipTest("Ignoring CUDA out of memory error!")

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
        self.assertFalse(tm.validation_loader.dataset.do_conditioning)
        # Calling __getitem__ on the dataloaders, we shouldn't have any condition tokens
        for dataloader in [tm.train_loader, tm.validation_loader]:
            for batch_idx in range(len(dataloader)):
                batch = dataloader.dataset.__getitem__(batch_idx)
                iids = batch["input_ids"].tolist()
                decoded = [tm.tokenizer[i] for i in iids]
                for tok in decoded:
                    self.assertFalse(tok.startswith(self.CONDITION_TOKENS))
        # Should not have a test dataloader
        self.assertTrue(tm.test_loader is None)
        self.assertTrue(len(self.PRETRAINER.track_splits["test"]) == 0)

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
        self.assertFalse(self.PRETRAINER.validation_loader.dataset.do_conditioning)
        # No test split
        self.assertTrue(self.PRETRAINER.test_loader is None)
        self.assertTrue(len(self.PRETRAINER.track_splits["test"]) == 0)


if __name__ == '__main__':
    unittest.main()
