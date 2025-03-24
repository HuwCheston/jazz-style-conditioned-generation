#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Training module"""

import os
from time import time

import mlflow
import numpy as np
import torch
import yaml
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Config, GPT2LMHeadModel

from jazz_style_conditioned_generation import utils, metrics
from jazz_style_conditioned_generation.data.dataloader import (
    DatasetMIDIConditionedRandomChunk,
    DatasetMIDIConditionedFullTrack,
    DATA_DIR
)
from jazz_style_conditioned_generation.data.tokenizer import (
    load_tokenizer,
    train_tokenizer,
    add_genres_to_vocab,
    add_pianists_to_vocab,
    add_tempos_to_vocab,
    add_timesignatures_to_vocab
)
from jazz_style_conditioned_generation.encoders import MusicTransformer, MusicTransformerScheduler
from jazz_style_conditioned_generation.preprocessing.splits import SPLIT_TYPES, SPLIT_DIR, check_all_splits_unique


class DummyModule(torch.nn.Module):
    """A dummy training module that simply returns an input tensor, for debugging"""

    def __init__(self, **_):
        super(DummyModule, self).__init__()
        # This means that we have a parameter
        self.param = torch.nn.Parameter(torch.ones(1))

    def forward(self, x: torch.tensor) -> torch.Tensor:
        return x


class DummyScheduler(torch.optim.lr_scheduler.LRScheduler):
    """An LR scheduler that does not modify anything but has the same API as all other schedulers."""

    def __init__(self, optimizer, last_epoch=-1):
        super(DummyScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Just returns the current learning rates without modification"""
        return [group['lr'] for group in self.optimizer.param_groups]


class TrainingModule:
    def __init__(
            self,
            experiment: str,
            run: str,
            batch_size: int,
            epochs: int,
            train_dataset_cfg: dict,
            test_dataset_cfg: dict,
            model_cfg: dict,
            optimizer_cfg: dict,
            scheduler_cfg: dict,
            checkpoint_cfg: dict,
            tokenizer_cfg: dict,
            mlflow_cfg: dict,
            generate_cfg: dict,
            data_dir: str = None,
            split_dir: str = None,
            full_validate_after_n_epochs: int = 25,
            n_full_validation_tracks: int = 10,
            _generate_only: bool = False
    ):
        # Set all keyword arguments to class parameters
        self.experiment = experiment
        self.run = run
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_dataset_cfg = train_dataset_cfg
        self.test_dataset_cfg = test_dataset_cfg
        self.model_cfg = model_cfg
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        self.checkpoint_cfg = checkpoint_cfg
        self.tokenizer_cfg = tokenizer_cfg
        self.mlflow_cfg = mlflow_cfg
        self.generate_cfg = generate_cfg
        self._data_dir = data_dir
        self._split_dir = split_dir
        self._generate_only = _generate_only  # should only be set to True when running generate.py

        # Initialise the current epoch at 0
        self.current_epoch = 0

        # DATA SPLITS
        self.track_splits = {split_type: list(self.read_tracks_for_split(split_type)) for split_type in SPLIT_TYPES}
        check_all_splits_unique(*list(self.track_splits.values()))
        logger.debug("Split tracks: " + ", ".join([f'{k}: {len(list(v))}' for k, v in self.track_splits.items()]))

        # MIDI PATHS
        self.track_paths = sorted([x for xs in self.track_splits.values() for x in xs])  # unpack to a flat list
        utils.validate_paths(self.track_paths, expected_extension=".mid")
        logger.debug(f"Loaded {len(self.track_paths)} tracks from {self.data_dir}")

        # METADATA PATHS
        self.metadata_paths = [fp.replace("piano_midi.mid", "metadata_tivo.json") for fp in self.track_paths]
        utils.validate_paths(self.metadata_paths, expected_extension=".json")
        logger.debug(f"Loaded {len(self.metadata_paths)} metadata JSONs from {self.data_dir}")

        # TOKENIZER
        self.tokenizer_path = os.path.join(self.checkpoint_dir, "tokenizer.json")
        self.tokenizer = load_tokenizer(tokenizer_path=self.tokenizer_path, **self.tokenizer_cfg)
        logger.debug(f'... tokenizer initialised: {self.tokenizer}')

        # CONDITIONS
        if self.train_dataset_cfg.get("do_conditioning", True) != self.test_dataset_cfg.get("do_conditioning", True):
            raise AttributeError('Got conflicting options for `do_conditioning` for test and train dataloaders!')
        # We want the conditioning tokens to always be part of the vocabulary
        # TODO: we probably don't want to do this when using a different condition type,
        #  e.g. concatenating along the sequence dimension
        logger.debug("Adding condition tokens...")
        # These functions add all the required condition tokens into the tokenizer's vocabulary
        add_genres_to_vocab(self.tokenizer)
        add_pianists_to_vocab(self.tokenizer)
        add_tempos_to_vocab(self.tokenizer, 80, 30, factor=1.05)
        add_timesignatures_to_vocab(self.tokenizer, [3, 4])
        # Log the number of tokens we've added for each condition type to the console
        for condition in ["GENRES", "PIANIST", "TIMESIGNATURE", "TEMPO"]:
            n_conditions = [i for i in self.tokenizer.vocab if i.startswith(condition)]
            logger.debug(f'... added {len(n_conditions)} {condition} tokens!')

        # TRAIN THE TOKENIZER
        if self.tokenizer_cfg.get("do_training", False):
            train_tokenizer(
                tokenizer=self.tokenizer,
                files_paths=self.track_paths,
                do_conditioning=self.train_dataset_cfg.get("do_conditioning", True),
                **self.tokenizer_cfg
            )
        # If we're not training, we need to add the condition tokens into the bpe_token_mapping attribute
        else:
            for tok in self.tokenizer.vocab.values():
                if tok not in self.tokenizer.bpe_token_mapping:
                    self.tokenizer.bpe_token_mapping[tok] = [tok]
            assert len(self.tokenizer.bpe_token_mapping) == len(self.tokenizer.vocab)

        # SAVE THE TOKENIZER (if it doesn't already exist)
        if not os.path.isfile(self.tokenizer_path):
            self.tokenizer.save(out_path=self.tokenizer_path)
            logger.debug(f"... dumped tokenizer to {self.tokenizer_path}")

        # DATALOADERS
        logger.debug(f'Initialising training loader with args {self.train_dataset_cfg}')
        logger.debug(f'Initialising testing + validation loaders with args {self.test_dataset_cfg}')
        self.train_loader, self.validation_loader, self.test_loader = self.create_dataloaders()

        # VALIDATION
        # After M epochs, we do a "full" validation with N complete tracks
        #  Note that we still validate with the full validation set after every epoch
        #  This just denotes how often we'll try and predict an ENTIRE track (rather than just a chunk)
        self.full_validate_after_n_epochs = full_validate_after_n_epochs
        self.n_full_validation_tracks = n_full_validation_tracks
        logger.debug(f"After completing {self.full_validate_after_n_epochs} epochs, "
                     f"we'll validate with {self.n_full_validation_tracks} tracks")

        # MODEL
        model_type = self.model_cfg.get("model_type", "gpt2-lm")
        model_kws = self.model_cfg.get("model_kws", dict())
        logger.debug(f'Initialising model {model_type} with arguments {model_kws}...')
        logger.debug(f"Training on device {utils.DEVICE}")
        self.model = self.get_model(model_type, model_kws).to(utils.DEVICE)
        logger.debug(f"Initialised model with {utils.total_parameters(self.model)} parameters")

        # LOSS & METRICS
        self.current_validation_loss = 0.
        self.best_validation_loss = 1e4  # should always be beaten...
        # TODO: consider multiple losses? what about a GAN loss too?

        # OPTIMISER
        optimizer_type = self.optimizer_cfg.get("optimizer_type", "adam")
        optimizer_kws = self.optimizer_cfg.get("optimizer_kws", dict(lr=0.0001))
        logger.debug(f'Initialising optimiser {optimizer_type} with parameters {optimizer_kws}...')
        self.optimizer = self.get_optimizer(optimizer_type)(self.model.parameters(), **optimizer_kws)

        # SCHEDULER
        self.sched_type = self.scheduler_cfg.get("scheduler_type", None)
        sched_kws = self.scheduler_cfg.get("scheduler_kws", dict())
        logger.debug(f'Initialising LR scheduler {self.sched_type} with parameters {sched_kws}...')
        self.scheduler = self.get_scheduler(self.sched_type, sched_kws)

        # EARLY STOPPING
        self.do_early_stopping = self.scheduler_cfg.get("do_early_stopping", False)
        self.min_lr = self.scheduler_cfg.get("min_lr", 1e-100)  # should never be reached by default

        # CHECKPOINTS
        if self.checkpoint_cfg.get("load_checkpoints", True):
            self.load_most_recent_checkpoint()

    def create_dataloaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Creates a dataloader for a given split and configuration"""
        # Create validation dataset loader: uses random chunks
        if self.test_dataset_cfg.get("do_augmentation", False):
            raise AttributeError("Augmentation only allowed for training dataloader!")
        # Create test dataset loader: uses FULL tracks!
        test_loader = DataLoader(
            DatasetMIDIConditionedFullTrack(
                tokenizer=self.tokenizer,
                files_paths=self.track_splits["test"],
                max_seq_len=utils.MAX_SEQUENCE_LENGTH,
                **self.test_dataset_cfg  # most arguments can be shared across test + validation loader
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
                **self.test_dataset_cfg  # most arguments can be shared across test + validation loader
            ),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )
        # Create test dataset loader: uses random chunks
        train_loader = DataLoader(
            DatasetMIDIConditionedRandomChunk(
                tokenizer=self.tokenizer,
                files_paths=self.track_splits["train"],
                max_seq_len=utils.MAX_SEQUENCE_LENGTH,
                **self.train_dataset_cfg
            ),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        return train_loader, validation_loader, test_loader

    def read_tracks_for_split(self, split_type: str) -> list[str]:
        """Reads a txt file containing a one line per string and returns as a list of strings"""
        split_fp = os.path.join(self.split_dir, split_type + '_split.txt')
        with open(split_fp, 'r') as fp:
            all_paths = fp.read().strip().split('\n')
            # Check that the path exists on the local file structure
            for path in all_paths:
                track_path = os.path.join(self.data_dir, path, "piano_midi.mid")
                if not os.path.isfile(track_path):
                    raise FileNotFoundError(f'Could not find MIDI for track at {track_path}')
                metadata_path = os.path.join(self.data_dir, path, "metadata_tivo.json")
                if not os.path.isfile(metadata_path):
                    raise FileNotFoundError(f'Could not find metadata for track at {metadata_path}')
                yield os.path.join(self.data_dir, path, "piano_midi.mid")

    def get_model(self, model_type: str, model_cfg: dict):
        """Given a string, returns the correct model"""
        valids = ["gpt2-lm", "music-transformer", None]
        # GPT-2 with language modelling head
        if model_type == "gpt2-lm":
            cfg = GPT2Config(
                vocab_size=self.tokenizer.vocab_size,
                n_positions=utils.MAX_SEQUENCE_LENGTH,
                bos_token_id=self.tokenizer["BOS_None"],
                eos_token_id=self.tokenizer["EOS_None"],
                **model_cfg
            )
            return GPT2LMHeadModel(cfg)
        # Music Transformer
        elif model_type == "music-transformer":
            return MusicTransformer(
                tokenizer=self.tokenizer,
                max_seq_len=utils.MAX_SEQUENCE_LENGTH,
                **model_cfg
            )
        # For debug purposes
        elif model_type is None:
            return DummyModule(**model_cfg)
        else:
            valid_types = ", ".join([i if i is not None else "None" for i in valids])
            raise ValueError(f'`model_type` must be one of {valid_types} but got {model_type}')

    @staticmethod
    def get_optimizer(optim_type: str):
        """Given a string, returns the correct optimizer"""
        valids = ["adam", "sgd"]
        if optim_type == "adam":
            return torch.optim.Adam
        elif optim_type == "sgd":
            return torch.optim.SGD
        else:
            raise ValueError(f'`optim_type` must be one of {", ".join(valids)} but got {optim_type}')

    def get_scheduler(self, sched_type: str | None, sched_kws: dict):
        """Given a string, returns the correct optimizer"""
        valids = ["plateau", "cosine", "step", "linear", "music-transformer", None]
        # This scheduler won't modify anything, but provides the same API for simplicity
        if sched_type is None:
            return DummyScheduler(self.optimizer, **sched_kws)
        elif sched_type == "reduce":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **sched_kws)
        elif sched_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **sched_kws)
        elif sched_type == "step":
            return torch.optim.lr_scheduler.StepLR(self.optimizer, **sched_kws)
        elif sched_type == "linear":
            return torch.optim.lr_scheduler.LinearLR(self.optimizer, **sched_kws)
        elif sched_type == "music-transformer":
            sched = MusicTransformerScheduler(**sched_kws)
            # This possibly won't work when resuming from a checkpoint?
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, sched.step)
        else:
            valid_types = ", ".join([i if i is not None else "None" for i in valids])
            raise ValueError(f'`sched_type` must be one of {valid_types} but got {sched_type}')

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load the checkpoint at the given fpath"""
        # This will raise a warning about possible ACE exploits, but we don't care
        try:
            loaded = torch.load(checkpoint_path, map_location=utils.DEVICE, weights_only=False)
        except FileNotFoundError:
            logger.error(f'Could not load checkpoint at {checkpoint_path}, skipping load!')
            return
        else:
            # Set state dictionary for all torch objects
            self.model.load_state_dict(loaded["model_state_dict"], strict=True)
            self.optimizer.load_state_dict(loaded["optimizer_state_dict"])
            # For backwards compatibility with no LR scheduler runs: don't worry if we can't load the LR scheduler dict
            try:
                self.scheduler.load_state_dict(loaded['scheduler_state_dict'])
            except KeyError:
                logger.warning("Could not find scheduler state dictionary in checkpoint, scheduler will be restarted!")
            # Increment epoch by 1
            self.current_epoch = loaded["epoch"] + 1
            self.scheduler.last_epoch = self.current_epoch
            # Set the current and best validation loss accordingly
            try:
                self.current_validation_loss = loaded["current_validation_loss"]
                self.best_validation_loss = loaded["best_validation_loss"]
            except KeyError:
                pass
            # We've moved saving a checkpoint to the end of the training loop, so we shouldn't need to do a step here
            # if self.sched_type == "reduce":
            #     self.scheduler.step(self.current_validation_loss)
            # else:
            #     self.scheduler.step()
            logger.debug(f'Loaded the checkpoint at {checkpoint_path}!')

    def load_most_recent_checkpoint(self) -> None:
        """Load the latest checkpoint for the current experiment and run"""
        # If we haven't created a checkpoint for this run, skip loading and train from scratch
        if not os.path.exists(self.checkpoint_dir):
            logger.warning('Checkpoint folder does not exist for this experiment/run, skipping load!')
            return

        # Get all the checkpoints for the current experiment/run combination, not including the "best" checkpoints
        checkpoints = [i for i in os.listdir(self.checkpoint_dir) if i.endswith(".pth") and "best" not in i]
        # Get the best validation loss checkpoint
        best_checkpoint = [i for i in os.listdir(self.checkpoint_dir) if i.endswith(".pth") and "best" in i]

        # We have no checkpoints, so we don't need to do anything
        if len(checkpoints) + len(best_checkpoint) == 0:
            logger.warning('No checkpoints have been created yet for this experiment/run, skipping load!')
            return

        # Otherwise, we need to compare the checkpoints
        # We have epoch checkpoints, but no best checkpoint
        elif len(checkpoints) > 0 and len(best_checkpoint) == 0:
            # Sort the checkpoints and load the latest one
            latest_checkpoint = sorted(checkpoints)[-1]
            checkpoint_path = os.path.join(self.checkpoint_dir, latest_checkpoint)

        # We only have a single, best checkpoint, so we'll just use this
        elif len(checkpoints) == 0 and len(best_checkpoint) > 0:
            checkpoint_path = os.path.join(self.checkpoint_dir, best_checkpoint[0])

        # We have both best checkpoint and most recent checkpoints
        else:
            # Sort the epoch checkpoints and load the most recent one
            latest_checkpoint = sorted(checkpoints)[-1]
            latest_check_path = os.path.join(self.checkpoint_dir, latest_checkpoint)
            latest_check_loaded = torch.load(latest_check_path, map_location=utils.DEVICE, weights_only=False)
            # Load the best checkpoint
            best_checkpoint_path = os.path.join(self.checkpoint_dir, best_checkpoint[0])
            best_checkpoint_loaded = torch.load(best_checkpoint_path, map_location=utils.DEVICE, weights_only=False)
            # If the best checkpoint is later than the most recent checkpoint, use this one
            if int(best_checkpoint_loaded["epoch"]) > int(latest_check_loaded["epoch"]):
                checkpoint_path = best_checkpoint_path
            # Otherwise, use the most recent checkpoint
            else:
                checkpoint_path = latest_check_path

        # Load the desired checkpoint
        self.load_checkpoint(os.path.join(self.checkpoint_dir, checkpoint_path))
        # Set a NEW random seed according to the epoch, otherwise we'll just use the same randomisations as epoch 1
        utils.seed_everything(utils.SEED * self.current_epoch)

    def save_checkpoint(self, epoch_metrics: dict, path: str) -> None:
        """Saves a checkpoint with given metrics to required path"""
        # Get the folder of checkpoints for the current experiment/run, and create if it doesn't exist
        run_folder = os.path.dirname(path)
        if not os.path.exists(run_folder):
            os.makedirs(run_folder, exist_ok=True)
        # Save everything, including the metrics, state dictionaries, and current epoch
        torch.save(
            dict(
                **epoch_metrics,
                model_state_dict=self.model.state_dict(),
                optimizer_state_dict=self.optimizer.state_dict(),
                scheduler_state_dict=self.scheduler.state_dict(),
                epoch=self.current_epoch
            ),
            os.path.join(path),
        )
        logger.debug(f'Saved a checkpoint to {run_folder}')

    @property
    def data_dir(self) -> str:
        if self._data_dir is not None:
            if utils.get_project_root() not in self._data_dir:
                self._data_dir = os.path.join(utils.get_project_root(), self._data_dir)
            assert os.path.isdir(self._data_dir), f"Data directory {self._data_dir} does not exist!"
            return self._data_dir
        else:
            return os.path.join(DATA_DIR, "raw")

    @property
    def split_dir(self) -> str:
        if self._split_dir is not None:
            if utils.get_project_root() not in self._split_dir:
                self._split_dir = os.path.join(utils.get_project_root(), self._split_dir)
            assert os.path.isdir(self._split_dir), f"Data directory {self._split_dir} does not exist!"
            return self._split_dir
        else:
            return SPLIT_DIR

    @property
    def checkpoint_dir(self) -> str:
        """Directory for saving model checkpoints, unique to this experiment and run"""
        # Either use a custom checkpoint directory or the root directory of the project (default)
        checkpoint_dir = self.checkpoint_cfg.get(
            "checkpoint_dir",
            os.path.join(utils.get_project_root(), 'checkpoints')
        )
        return os.path.join(checkpoint_dir, self.experiment, self.run)

    @property
    def output_midi_dir(self) -> str:
        """Directory for saving generated MIDI outputs, unique to this experiment and run"""
        outputs_dir = os.path.join(utils.get_project_root(), "outputs/generation", self.experiment, self.run)
        if not os.path.isdir(outputs_dir):
            os.makedirs(outputs_dir, exist_ok=True)
        return outputs_dir

    def remove_old_checkpoints(self):
        """Removes old checkpoints from the hard disk"""
        logger.debug('Deleting old checkpoints...')
        # Get the name of the current checkpoint
        new_check_name = f'checkpoint_{str(self.current_epoch).zfill(3)}.pth'
        # Iterate through all the checkpoints in our folder
        for old_checkpoint in os.listdir(self.checkpoint_dir):
            # Skip over deleting our best runs!
            if old_checkpoint.endswith('_best.pth'):
                continue
            # If the checkpoint is different to the most recent one and is a valid checkpoint
            if old_checkpoint != new_check_name and old_checkpoint.endswith('.pth'):
                # Remove the old checkpoint
                old_path = os.path.join(self.checkpoint_dir, old_checkpoint)
                os.remove(old_path)
                logger.info(f'... deleted {old_path}')

    def get_scheduler_lr(self) -> float:
        """Tries to return current LR from scheduler. Returns 0. on error, for safety with MLFlow logging"""
        try:
            return self.scheduler.get_last_lr()[0]
        except (IndexError, AttributeError, RuntimeError, TypeError) as sched_e:
            logger.warning(f"Failed to get LR from scheduler! Returning 0.0... {sched_e}")
            return 0.

    def step(self, batch: dict[str, torch.tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = batch["input_ids"].to(utils.DEVICE)
        labels = batch["labels"].to(utils.DEVICE)
        attention_mask = batch["attention_mask"].to(utils.DEVICE)
        logits = self.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
        loss = metrics.cross_entropy_loss(logits, labels, self.tokenizer)
        accuracy = metrics.accuracy_score(logits, labels, self.tokenizer)
        return loss, accuracy

    def training(self, epoch_num: int) -> tuple[float, float]:
        self.model.train()
        epoch_loss, epoch_accuracy = [], []
        # Iterate over every batch in the dataloader
        for batch_idx, batch in tqdm(
                enumerate(self.train_loader),
                total=len(self.train_loader),
                desc=f'Training, epoch {epoch_num} / {self.epochs}...'
        ):
            # Forwards pass
            loss, accuracy = self.step(batch)
            # Backwards pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # Append metrics to the list
            epoch_loss.append(loss.item())
            epoch_accuracy.append(accuracy.item())
        return np.mean(epoch_loss), np.mean(epoch_accuracy)

    def remove_condition_tokens(self, tensor: torch.tensor) -> torch.tensor:
        """Removes conditioning tokens from a tensor"""
        # Copy the tensor so we don't overwrite it
        new_tensor = tensor.detach().clone()
        # Iterate through all the condition token IDs
        for id_ in self.tokenizer.special_tokens_ids[4:]:
            # Replace them with the pad token ID
            new_tensor[new_tensor == id_] = self.tokenizer.pad_token_id
        return new_tensor

    def generate_from_batch(self, batch: dict, stage: str) -> None:
        """Given a batch of items, generates a random example from one of the items"""
        # Get the index of the element we're going to use to generate from
        generate_idx = torch.randint(batch["input_ids"].size(0), (1,)).item()
        # Get the input IDs and attention mask from the corresponding element
        gen_iid = batch["input_ids"][generate_idx].unsqueeze(0)
        gen_am = batch["attention_mask"][generate_idx].unsqueeze(0)
        # Generate using the model
        gen = self.model.generate(
            gen_iid.to(utils.DEVICE),
            attention_mask=gen_am.to(utils.DEVICE),
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer["BOS_None"],
            eos_token_id=self.tokenizer["EOS_None"],
            bad_words_ids=[self.tokenizer.special_tokens_ids[4:]],  # prevent model from generating conditioning tokens
            renormalize_logits=True,  # documentation says "highly recommended" to set this to True
            **self.generate_cfg.get("generate_kws", dict())
        )
        try:
            # Replace any generated condition tokens with pad tokens, then decode with the tokenizer
            outs = self.tokenizer.decode(self.remove_condition_tokens(gen.to('cpu')))
            inds = self.tokenizer.decode(self.remove_condition_tokens(gen_iid.to('cpu')))
        except KeyError:
            logger.warning(f"Couldn't generate from batch with tokens {gen_iid.tolist()}")
        else:
            # Dump everything to the output directory for this experiment/run
            now = utils.now()
            inds.dump_midi(f"{self.output_midi_dir}/input_{stage}_epoch{self.current_epoch}_{now}.mid")
            outs.dump_midi(f"{self.output_midi_dir}/output_{stage}_epoch{self.current_epoch}_{now}.mid")
            logger.info(f"Dumped {stage} MIDI from epoch {self.current_epoch} to {self.output_midi_dir}")

    def validation(self, epoch_num: int) -> tuple[float, float]:
        self.model.eval()
        epoch_loss, epoch_accuracy = [], []
        # Iterate over every batch in the dataloader
        for batch_idx, batch in tqdm(
                enumerate(self.validation_loader),
                total=len(self.validation_loader),
                desc=f'Validating, epoch {epoch_num} / {self.epochs}...'
        ):
            # Forwards pass
            with torch.no_grad():
                loss, accuracy = self.step(batch)
            # No backwards pass
            epoch_loss.append(loss.item())
            epoch_accuracy.append(accuracy.item())
            # Generate from this batch if required
            if self.generate_cfg.get("do_generation", True):
                if utils.random_probability() < self.generate_cfg.get("generation_probability", 0.01):
                    self.generate_from_batch(batch, "validation")
        return np.mean(epoch_loss), np.mean(epoch_accuracy)

    def testing(self) -> tuple[float, float]:
        # Load the checkpoint with the best validation loss
        if self.checkpoint_cfg.get("load_checkpoints", True):
            self.load_checkpoint(os.path.join(self.checkpoint_dir, 'validation_best.pth'))
        # Run the evaluate_full_tracks function on the ENTIRE test dataset
        full_test_loss = self.evaluate_full_tracks(len(self.test_loader))
        return full_test_loss

    def evaluate_full_tracks(self, n_full_tracks: int = None):
        """Tests on full dataset of N test tracks and computes loss that should be comparable across vocab sizes"""
        if n_full_tracks is None:
            n_full_tracks = self.n_full_validation_tracks
        self.model.eval()
        full_track_losses = []
        for batch_idx, batch in enumerate(self.test_loader):
            # Break out once we've considered enough tracks
            #  These tracks should always be identical between runs as we set shuffle=False in the dataloader
            if batch_idx > n_full_tracks:
                break
            logger.info(f'Processing track {batch_idx + 1} / {n_full_tracks} ...')
            full_track_loss = self.model.evaluate(
                batch["input_ids"].to(utils.DEVICE),
                batch["labels"].to(utils.DEVICE),
                batch["attention_mask"].to(utils.DEVICE),
                batch_size=self.batch_size
            )
            full_track_losses.append(full_track_loss.item())
        return np.mean(full_track_losses)

    def log_run_params_to_mlflow(self):
        """If we're using MLFlow, log all run parameters to the dashboard"""
        # These types are valid types for logging (i.e., we don't want to log a list or dictionary...)
        valid_log_types = (str, float, int, bool)

        def logme(k, v):
            try:
                mlflow.log_param(k, v)
            except mlflow.exceptions.MlflowException:
                logger.warning(f'Unable to log param {k} with value {v} to dashboard: '
                               f'has it already been logged (if this run is resumed from a checkpoint)?')

        for key, val in self.__dict__.items():
            if isinstance(val, valid_log_types):
                logme(key, val)
            elif isinstance(val, dict):
                for inner_key, inner_val in val.items():
                    if isinstance(inner_val, (str, float, int, bool)):
                        logme(f'{key}_{inner_key}', inner_val)
            else:
                continue

        logger.debug("Logged all run parameters to MLFlow dashboard!")

    def start(self):
        """Runs training for this module"""
        training_start = time()
        # Log parameters for the run to MLflow if required
        if self.mlflow_cfg.get("use", False):
            self.log_run_params_to_mlflow()
        # Start training
        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch
            epoch_start = time()
            # If required, stop early once we've reached the minimum LR
            if self.do_early_stopping and self.scheduler.get_last_lr()[-1] <= self.min_lr:
                logger.warning(f"Early stopping! LR {self.scheduler.get_last_lr()[-1]} reached {self.min_lr}")
                break
            # Training
            train_loss, train_accuracy = self.training(epoch)
            logger.debug(f'Epoch {epoch} / {self.epochs}, training finished: '
                         f'loss {train_loss:.3f}, accuracy {train_accuracy:.3f}')
            # Validation
            self.current_validation_loss, validation_accuracy = self.validation(epoch)
            logger.debug(f'Epoch {epoch} / {self.epochs}, validation finished: '
                         f'loss {self.current_validation_loss:.3f}, accuracy {validation_accuracy:.3f}')
            # Log if this is our best epoch
            if self.current_validation_loss < self.best_validation_loss:
                self.best_validation_loss = self.current_validation_loss
                logger.info(f'New best validation loss: {self.current_validation_loss:.3f}')
            # Log parameters from this epoch in MLFlow
            epoch_metrics = dict(
                epoch_time=time() - epoch_start,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                current_validation_loss=self.current_validation_loss,
                best_validation_loss=self.best_validation_loss,
                validation_accuracy=validation_accuracy,
                lr=self.get_scheduler_lr()
            )
            # Run evaluation on 10 full tracks every few epochs
            if epoch % self.full_validate_after_n_epochs == 0:
                logger.debug(f"Doing a partial validation with {self.n_full_validation_tracks} complete tracks...")
                valid_loss_full_track = self.evaluate_full_tracks(self.n_full_validation_tracks)
                logger.info(f"Epoch {epoch} / {self.epochs}: full-track validation loss {valid_loss_full_track:.3f}")
                epoch_metrics["validation_loss_full_track"] = valid_loss_full_track
            # Report results to MLFlow, if we're using this
            if self.mlflow_cfg.get("use", False):
                mlflow.log_metrics(epoch_metrics, step=epoch)
            # Step forward in the LR scheduler
            # ReduceLROnPlateau needs the current validation loss passed in
            if self.sched_type == "reduce":
                self.scheduler.step(self.current_validation_loss)
            # Other schedulers don't require anything
            else:
                self.scheduler.step()
            logger.debug(f'LR for epoch {epoch + 1} will be {self.get_scheduler_lr()}')
            # Checkpoint the run, if we need to
            if self.checkpoint_cfg["save_checkpoints"]:
                # How many epochs before we need to checkpoint (10 by default)
                checkpoint_after = self.checkpoint_cfg.get("checkpoint_after_n_epochs", 10)
                # The name of the checkpoint and where it'll be saved
                new_check_name = f'checkpoint_{str(self.current_epoch).zfill(3)}.pth'
                # We always want to checkpoint on the final epoch!
                if (self.current_epoch % checkpoint_after == 0) or (self.current_epoch + 1 == self.epochs):
                    self.save_checkpoint(epoch_metrics, os.path.join(self.checkpoint_dir, new_check_name))
                # If we want to remove old checkpoints after saving a new one
                if self.checkpoint_cfg.get("delete_old_checkpoints", False):
                    self.remove_old_checkpoints()
                # Save an additional checkpoint for the run if this is the best epoch
                if self.current_validation_loss == self.best_validation_loss:
                    self.save_checkpoint(epoch_metrics, os.path.join(self.checkpoint_dir, 'validation_best.pth'))
        # Run testing after training completes
        logger.info('Training complete!')
        test_loss_full_track = self.testing()
        # Report results to MLFlow, if we're using this
        if self.mlflow_cfg.get("use", False):
            test_metrics = dict(test_loss_full_track=test_loss_full_track)
            mlflow.log_metrics(test_metrics, step=self.current_epoch)
        # Log everything to the console
        logger.info(f"Testing finished: full-track loss {test_loss_full_track:.3f}")
        logger.info(f'Finished in {(time() - training_start) // 60} minutes!')


def parse_config_yaml(fpath: str) -> dict:
    """Parses a configuration YAML file at `fpath`"""
    full_fpath = os.path.join(utils.get_project_root(), 'config', fpath)
    if not full_fpath.endswith(".yaml"):
        full_fpath += ".yaml"
    try:
        with open(full_fpath) as stream:
            cfg = yaml.safe_load(stream)
    except FileNotFoundError:
        raise FileNotFoundError(f'Config file {full_fpath} could not be found')
    else:
        logger.info(f'Config file {full_fpath} parsed')
        return cfg


def get_tracking_uri(port: str = "8080") -> str:
    """Attempts to get the MLflow tracking URI on given port based on system hostname"""
    import socket

    hostname = socket.gethostname().lower()
    # Job is running locally: don't use mlflow
    if "desktop" in hostname:
        return None
    # Job is running on the department server
    elif "musix" in hostname:
        return f"http://127.0.0.1:{port}"
    # Job is running on HPC
    else:
        return f"http://musix.mus.cam.ac.uk:{port}"


def add_run_id_to_config_yaml(config_fname: str, mlflow_run_id: str) -> None:
    """Append an automatically-created mlflow run ID to a config `.yaml` file at the start of a new run"""
    # This is the directory where our config file is
    yamlpath = os.path.join(utils.get_project_root(), 'config', config_fname)
    # Load the config file
    with open(yamlpath, 'r') as yamlfile:
        cur_yaml = yaml.safe_load(yamlfile)
        # Create a new mlflow dictionary, if for whatever reason we don't have this
        if 'mlflow_cfg' not in cur_yaml.keys():
            cur_yaml['mlflow_cfg'] = {}
        # Add the run ID into our config file
        cur_yaml['mlflow_cfg']['run_id'] = mlflow_run_id
    # Overwrite the config file, without sorting the keys
    with open(yamlpath, 'w') as yamlfile:
        yaml.safe_dump(cur_yaml, yamlfile, sort_keys=False)


def main(training_kws: dict, trainer_cls: type = TrainingModule, config_fpath: str = None) -> None:
    """
    Runs training with given kwargs.

    trainer_cls should be a class that implements e.g .start, .step methods. config_fpath should be a path towards
    a config .yaml file that will be parsed

    """

    # Running training with logging on MLFlow
    if training_kws["mlflow_cfg"]["use"]:
        # Get the run ID from our config. Fall back to None if not provided
        run_id = training_kws["mlflow_cfg"].get("run_id", None)
        # Get the tracking URI based on the hostname of the device running the job
        uri = get_tracking_uri(port=training_kws["mlflow_cfg"].get("port", "8080"))
        if uri is None:
            raise ValueError(f'Could not connect to MLFlow!')
        else:
            logger.debug(f'Attempting to connect to MLFlow server at {uri}...')
            mlflow.set_tracking_uri(uri=uri)
            try:
                mlflow.set_experiment(training_kws["experiment"])
            # If we're unable to reach the MLFlow server somehow
            except mlflow.exceptions.MlflowException as err:
                logger.warning(f'Could not connect to MLFlow, falling back to running locally! {err}')
                # This will mean we break out of the current IF statement, and activate the next IF NOT statement
                # in order to train locally, without using MLFlow
                training_kws["mlflow_cfg"]["use"] = False
            else:
                # Otherwise, start training with the arguments we've passed in
                tm = trainer_cls(**training_kws)
                # Either run is being resumed with a run ID passed in with our config file
                if run_id is not None:
                    logger.debug(f'Resuming run with name {training_kws["run"]}, ID {run_id}!')
                # Or this is a new run
                else:
                    logger.debug(f'Starting new run with name {training_kws["run"]}!')
                # Start the run!
                with mlflow.start_run(run_name=training_kws["run"], run_id=run_id):
                    # If this is a new run, append the newly-created run ID to our yaml config file (if we passed this)
                    if config_fpath is not None and 'run_id' not in training_kws['mlflow_cfg'].keys():
                        new_run_id = mlflow.active_run().info.run_id
                        add_run_id_to_config_yaml(config_fpath, new_run_id)
                        logger.debug(f'Added run id {new_run_id} to {config_fpath}!')
                    tm.start()

    # Running training locally
    else:
        tm = trainer_cls(**training_kws)
        tm.start()


if __name__ == "__main__":
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
    training_kwargs = parse_config_yaml(parser_args['config'])
    training_kwargs["_generate_only"] = False  # should only be set to True when running generate.py
    # Start training!
    main(training_kwargs, config_fpath=parser_args["config"])
