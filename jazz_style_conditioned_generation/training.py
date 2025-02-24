#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Training module"""

import os
from pathlib import Path
from time import time

import mlflow
import numpy as np
import torch
from loguru import logger
from miditok.pytorch_data import DataCollator
from miditok.utils import split_files_for_training
from tqdm import tqdm
from transformers import GPT2Config, GPT2LMHeadModel

from jazz_style_conditioned_generation import utils
from jazz_style_conditioned_generation.data import (
    validate_conditions,
    get_mapping_for_condition,
    get_special_tokens_for_condition,
    DATA_DIR,
    DatasetMIDICondition,
    get_tokenizer,
    DEFAULT_TOKENIZER_CONFIG,
    DEFAULT_TOKENIZER_CLASS,
    DEFAULT_TRAINING_METHOD,
    SPLIT_DIR,
    SPLIT_TYPES,
    check_all_splits_unique
)


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
            conditions: list[int],
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
            generate_cfg: dict
    ):
        # Set all keyword arguments to class parameters
        self.experiment = experiment
        self.run = run
        self.conditions = conditions
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

        # Initialise the current epoch at 0
        self.current_epoch = 0

        # CONDITIONS
        validate_conditions(self.conditions)
        # this maps e.g. {"genre": {"African Jazz": 0, "African Folk": 1}, "moods": {"Aggressive": 0}, ...}
        self.condition_mapping = {c: get_mapping_for_condition(c) for c in self.conditions}
        logger.debug(f"Using conditions: " + ", ".join(self.conditions))
        n_unique_conditions = sum(len(list(v)) for v in self.condition_mapping.values())
        logger.debug(
            "Unique conditions: " + ", ".join([f'{k}: {len(list(v))}' for k, v in self.condition_mapping.items()])
        )

        # TOKENIZER
        tokenizer_method = self.tokenizer_cfg.get("tokenizer_str", DEFAULT_TOKENIZER_CLASS)
        training_method = self.tokenizer_cfg.get("training_method", DEFAULT_TRAINING_METHOD)
        tokenizer_kws = self.tokenizer_cfg.get("tokenizer_kws", DEFAULT_TOKENIZER_CONFIG)
        tokenizer_kws = utils.update_dictionary(tokenizer_kws, DEFAULT_TOKENIZER_CONFIG)
        logger.debug(
            f"Initialising tokenizer with method {tokenizer_method}, "
            f"training {training_method}, parameters {tokenizer_kws}"
        )
        self.tokenizer = get_tokenizer(tokenizer_method, training_method, tokenizer_kws)

        # SPECIAL TOKENS
        sts = [get_special_tokens_for_condition(c, self.condition_mapping[c]) for c in self.conditions]
        # This is a list of e.g. ["GENRE_0", "GENRE_1", "GENRE_2", ...]
        self.special_tokens = [x for xs in sts for x in xs]
        assert len(self.special_tokens) == n_unique_conditions
        logger.debug(f"Got {len(self.special_tokens)} condition tokens")
        # Add special tokens into tokenizer vocabulary
        for st in self.special_tokens:
            self.tokenizer.add_to_vocab(st, special_token=True)

        # DATA SPLITS
        self.track_splits = {split_type: list(self.read_tracks_for_split(split_type)) for split_type in SPLIT_TYPES}
        check_all_splits_unique(*list(self.track_splits.values()))
        self.track_paths = sorted([x for xs in self.track_splits.values() for x in xs])  # unpack to a flat list
        logger.debug(f"Loaded {len(self.track_paths)} tracks from {os.path.join(DATA_DIR, 'raw')}")
        logger.debug("Split tracks: " + ", ".join([f'{k}: {len(list(v))}' for k, v in self.track_splits.items()]))

        # DATA CHUNKING
        self.chunk_paths = self.get_midi_chunks()
        self.chunk_splits = self.chunk_paths_to_splits(self.track_splits, self.chunk_paths)
        check_all_splits_unique(*list(self.chunk_splits.values()))
        logger.debug(f"Loaded {len(self.chunk_paths)} MIDI chunks from {os.path.join(DATA_DIR, 'chunks')}")
        logger.debug("Split chunks: " + ", ".join([f'{k}: {len(list(v))}' for k, v in self.chunk_splits.items()]))

        # DATALOADERS
        logger.debug(f'Initialising training loader with args {self.train_dataset_cfg}')
        self.train_loader = self.create_dataloader("train", self.train_dataset_cfg)
        self.test_loader = self.create_dataloader("test", self.test_dataset_cfg)
        logger.debug(f'Initialising testing + validation loaders with args {self.test_dataset_cfg}')
        self.validation_loader = self.create_dataloader("validation", self.test_dataset_cfg)  # reuse test config

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
        sched_type = self.scheduler_cfg.get("scheduler_type", None)
        sched_kws = self.scheduler_cfg.get("scheduler_kws", dict())
        logger.debug(f'Initialising LR scheduler {sched_type} with parameters {sched_kws}...')
        self.scheduler = self.get_scheduler(sched_type)(self.optimizer, **sched_kws)

        # CHECKPOINTS
        if self.checkpoint_cfg.get("load_checkpoints", True):
            self.load_most_recent_checkpoint()

    def create_dataloader(self, split: str, dataset_cfg: dict) -> torch.utils.data.DataLoader:
        """Creates a dataloader for a given split and configuration"""
        dataset = DatasetMIDICondition(
            condition_mapping=self.condition_mapping,
            files_paths=self.chunk_splits[split],
            tokenizer=self.tokenizer,
            max_seq_len=utils.MAX_SEQUENCE_LENGTH,
            bos_token_id=self.tokenizer["BOS_None"],
            eos_token_id=self.tokenizer["EOS_None"],
            **dataset_cfg
        )
        collator = DataCollator(
            self.tokenizer.pad_token_id,
            copy_inputs_as_labels=True,
            shift_labels=True

        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=collator,
            shuffle=True,
            drop_last=False,
        )

    @staticmethod
    def read_tracks_for_split(split_type: str) -> list[str]:
        """Reads a txt file containing a one line per string and returns as a list of strings"""
        split_fp = os.path.join(SPLIT_DIR, split_type + '_split.txt')
        with open(split_fp, 'r') as fp:
            all_paths = fp.read().strip().split('\n')
            # Check that the path exists on the local file structure
            for path in all_paths:
                if not os.path.isfile(os.path.join(DATA_DIR, "raw", path, "piano_midi.mid")):
                    raise FileNotFoundError(f'Could not find MIDI for track at {path}')
                if not os.path.isfile(os.path.join(DATA_DIR, "raw", path, "metadata_tivo.json")):
                    raise FileNotFoundError(f'Could not find metadata for track at {path}')
                yield os.path.join(DATA_DIR, "raw", path, "piano_midi.mid")

    def get_model(self, model_type: str, model_cfg: dict):
        """Given a string, returns the correct model"""
        valids = ["gpt2-lm", None]
        # GPT-2 with language modelling head
        if model_type == "gpt2-lm":
            cfg = GPT2Config(
                vocab_size=self.tokenizer.vocab_size,
                n_positions=utils.MAX_SEQUENCE_LENGTH * 2,  # TODO: FIX THIS!
                bos_token_id=self.tokenizer["BOS_None"],
                eos_token_id=self.tokenizer["EOS_None"],
                **model_cfg
            )
            return GPT2LMHeadModel(cfg)
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

    @staticmethod
    def get_scheduler(sched_type: str | None):
        """Given a string, returns the correct optimizer"""
        valids = ["plateau", "cosine", "step", "linear", None]
        # This scheduler won't modify anything, but provides the same API for simplicity
        if sched_type is None:
            return DummyScheduler
        elif sched_type == "reduce":
            return torch.optim.lr_scheduler.ReduceLROnPlateau
        elif sched_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR
        elif sched_type == "step":
            return torch.optim.lr_scheduler.StepLR
        elif sched_type == "linear":
            return torch.optim.lr_scheduler.LinearLR
        else:
            valid_types = ", ".join([i if i is not None else "None" for i in valids])
            raise ValueError(f'`sched_type` must be one of {valid_types} but got {sched_type}')

    def get_midi_chunks(self) -> list[str]:
        """Split full MIDI tracks into chunks for training using MIDITok"""
        # Define file paths: MIDITok requires these to be pathlib.Path objects, not strings
        dataset_chunks_dir = Path(os.path.join(DATA_DIR, "chunks"))
        logger.debug(f"MIDI chunks will be stored at {str(dataset_chunks_dir)}")
        files_paths = sorted([Path(t) for t in self.track_paths])
        # This will pull from a cache if we've already chunked these files
        try:
            env = os.environ["PYTHONHASHSEED"]
        except KeyError:
            logger.warning('Hash randomization is enabled! MIDITok will not pull cached chunks correctly! '
                           'Pass PYTHONHASHSEED=0 as environment variable.')
        else:
            if env != "0":
                logger.warning('Hash randomization is enabled! MIDITok will not pull cached chunks correctly! '
                               'Pass PYTHONHASHSEED=0 as environment variable.')

        chunk_paths = split_files_for_training(
            files_paths=files_paths,
            tokenizer=self.tokenizer,
            save_dir=dataset_chunks_dir,
            max_seq_len=utils.MAX_SEQUENCE_LENGTH,  # TODO: this should be MAX_SEQUENCE_LENGTH - MAX_CONDITION_TOKENS?
            # This is important: it ensures that two chunks overlap slightly, to allow a causal chain between the
            #  end of one chunk and the beginning of the next
            num_overlap_bars=utils.CHUNK_OVERLAP_BARS
        )
        return sorted([str(cp) for cp in chunk_paths])

    @staticmethod
    def chunk_paths_to_splits(track_splits: dict[str, list[str]], chunk_paths: list[str]) -> dict[str, list[str]]:
        """For any MIDI track, we can have multiple chunks. We need to parse these into the original splits"""
        chunk_splits = {sp: [] for sp in SPLIT_TYPES}
        for chunk in chunk_paths:
            chunk_dirname = os.path.join(os.path.dirname(chunk).replace('chunks', 'raw'), "piano_midi.mid")
            for split_type in SPLIT_TYPES:
                if chunk_dirname in track_splits[split_type]:
                    chunk_splits[split_type].append(chunk)
        # We should have exactly the same number of MIDI chunks that we started with
        assert sum(len(list(v)) for v in chunk_splits.values()) == len(chunk_paths)
        return chunk_splits

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
            # For some reason, we need to do a step in the scheduler here so that we have the correct LR
            self.scheduler.step()
            logger.debug(f'Loaded the checkpoint at {checkpoint_path}!')

    def load_most_recent_checkpoint(self) -> None:
        """Load the latest checkpoint for the current experiment and run"""
        # If we haven't created a checkpoint for this run, skip loading and train from scratch
        if not os.path.exists(self.checkpoint_dir):
            logger.warning('Checkpoint folder does not exist for this experiment/run, skipping load!')
            return

        # Get all the checkpoints for the current experiment/run combination, not including the "best" checkpoints
        checkpoints = [i for i in os.listdir(self.checkpoint_dir) if i.endswith(".pth") and "best" not in i]
        if len(checkpoints) == 0:
            logger.warning('No checkpoints have been created yet for this experiment/run, skipping load!')
            return
        # Sort the checkpoints and load the latest one
        latest_checkpoint = sorted(checkpoints)[-1]
        checkpoint_path = os.path.join(self.checkpoint_dir, latest_checkpoint)
        self.load_checkpoint(os.path.join(self.checkpoint_dir, checkpoint_path))
        # Set a NEW random seed according to the epoch, otherwise we'll just use the same randomisations as epoch 1
        utils.seed_everything(utils.SEED * self.current_epoch)

    def save_checkpoint(self, metrics: dict, path: str) -> None:
        """Saves a checkpoint with given metrics to required path"""
        # Get the folder of checkpoints for the current experiment/run, and create if it doesn't exist
        run_folder = os.path.dirname(path)
        if not os.path.exists(run_folder):
            os.makedirs(run_folder, exist_ok=True)
        # Save everything, including the metrics, state dictionaries, and current epoch
        torch.save(
            dict(
                **metrics,
                model_state_dict=self.model.state_dict(),
                optimizer_state_dict=self.optimizer.state_dict(),
                scheduler_state_dict=self.scheduler.state_dict(),
                epoch=self.current_epoch
            ),
            os.path.join(path),
        )
        logger.debug(f'Saved a checkpoint to {run_folder}')

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
        except (IndexError, AttributeError, RuntimeError) as sched_e:
            logger.warning(f"Failed to get LR from scheduler! Returning 0.0... {sched_e}")
            return 0.

    def step(self, batch: dict[str, torch.tensor]) -> torch.tensor:
        input_ids = batch["input_ids"].to(utils.DEVICE)
        labels = batch["labels"].to(utils.DEVICE)
        attention_mask = batch["attention_mask"].to(utils.DEVICE)
        outputs = self.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
        return outputs.loss

    def training(self, epoch_num: int) -> float:
        self.model.train()
        epoch_loss = []
        # Iterate over every batch in the dataloader
        for batch in tqdm(
                self.train_loader,
                total=len(self.train_loader),
                desc=f'Training, epoch {epoch_num} / {self.epochs}...'
        ):
            # TODO: data augmentation?
            # Forwards pass
            loss = self.step(batch)
            # Backwards pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # Append metrics to the list
            epoch_loss.append(loss.item())
        return np.mean(epoch_loss)

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

    def validation(self, epoch_num: int) -> float:
        self.model.eval()
        epoch_loss = []
        # Iterate over every batch in the dataloader
        for batch in tqdm(
                self.validation_loader,
                total=len(self.validation_loader),
                desc=f'Validating, epoch {epoch_num} / {self.epochs}...'
        ):
            # Forwards pass
            with torch.no_grad():
                loss = self.step(batch)
            # No backwards pass
            epoch_loss.append(loss.item())
            # Generate from this batch if required
            if self.generate_cfg.get("do_generation", True):
                if utils.random_probability() < self.generate_cfg.get("generation_probability", 0.01):
                    self.generate_from_batch(batch, "validation")
        return np.mean(epoch_loss)

    def testing(self) -> float:
        # Load the checkpoint with the best validation loss
        if self.checkpoint_cfg.get("load_checkpoints", True):
            self.load_checkpoint(os.path.join(self.checkpoint_dir, 'validation_best.pth'))
        self.model.eval()
        epoch_loss = []
        # Iterate over every batch in the dataloader
        for batch in tqdm(
                self.test_loader,
                total=len(self.test_loader),
                desc='Testing...'
        ):
            # Forwards pass
            with torch.no_grad():
                loss = self.step(batch)
            # No backwards pass
            epoch_loss.append(loss.item())
            # Generate from this batch if required
            if self.generate_cfg.get("do_generation", True):
                if utils.random_probability() < self.generate_cfg.get("generation_probability", 0.01):
                    self.generate_from_batch(batch, "testing")
        return np.mean(epoch_loss)

    def log_run_params_to_mlflow(self):
        """If we're using MLFlow, log all run parameters to the dashboard"""

        def logme(k, v):
            try:
                mlflow.log_param(k, v)
            except mlflow.exceptions.MlflowException:
                logger.warning(f'Unable to log param {k} with value {v} to dashboard: '
                               f'has it already been logged (if this run is resumed from a checkpoint)?')

        for key, val in self.__dict__.items():
            if isinstance(val, (str, float, int, bool)):
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
            # Training
            train_loss = self.training(epoch)
            logger.debug(f'Epoch {epoch} / {self.epochs}, training finished: loss {train_loss:.3f}')
            # Validation
            self.current_validation_loss = self.validation(epoch)
            logger.debug(f'Epoch {epoch} / {self.epochs}, validation finished: loss {self.current_validation_loss:.3f}')
            # Log if this is our best epoch
            if self.current_validation_loss < self.best_validation_loss:
                self.best_validation_loss = self.current_validation_loss
                logger.info(f'New best validation loss: {self.current_validation_loss:.3f}')
            # Log parameters from this epoch in MLFlow
            metrics = dict(
                epoch_time=time() - epoch_start,
                train_loss=train_loss,
                current_validation_loss=self.current_validation_loss,
                best_validation_loss=self.best_validation_loss,
                lr=self.get_scheduler_lr()
            )
            # Checkpoint the run, if we need to
            if self.checkpoint_cfg["save_checkpoints"]:
                # How many epochs before we need to checkpoint (10 by default)
                checkpoint_after = self.checkpoint_cfg.get("checkpoint_after_n_epochs", 10)
                # The name of the checkpoint and where it'll be saved
                new_check_name = f'checkpoint_{str(self.current_epoch).zfill(3)}.pth'
                # We always want to checkpoint on the final epoch!
                if (self.current_epoch % checkpoint_after == 0) or (self.current_epoch + 1 == self.epochs):
                    self.save_checkpoint(metrics, os.path.join(self.checkpoint_dir, new_check_name))
                # If we want to remove old checkpoints after saving a new one
                if self.checkpoint_cfg.get("delete_old_checkpoints", False):
                    self.remove_old_checkpoints()
                # Save an additional checkpoint for the run if this is the best epoch
                if self.current_validation_loss == self.best_validation_loss:
                    self.save_checkpoint(metrics, os.path.join(self.checkpoint_dir, 'validation_best.pth'))
            # Report results to MLFlow, if we're using this
            if self.mlflow_cfg.get("use", False):
                mlflow.log_metrics(metrics, step=epoch)
            # Step forward in the LR scheduler
            self.scheduler.step()
            logger.debug(f'LR for epoch {epoch + 1} will be {self.get_scheduler_lr()}')
        # Run testing after training completes
        logger.info('Training complete!')
        test_loss = self.testing()
        logger.info(f"Testing finished: loss {test_loss:.3f}")
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


if __name__ == "__main__":
    import argparse
    import yaml

    # Seed everything for reproducible results
    utils.seed_everything(utils.SEED)

    # Parsing arguments from the command line interface
    parser = argparse.ArgumentParser(description="Run model training")
    parser.add_argument("-c", "--config", default=None, type=str, help="Path to config YAML file")
    # Parse all arguments from the provided YAML file
    args = vars(parser.parse_args())
    if not args:
        raise ValueError("No config file specified")
    training_kws = parse_config_yaml(args['config'])

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
                tm = TrainingModule(**training_kws)
                # Either run is being resumed with a run ID passed in with our config file
                if run_id is not None:
                    logger.debug(f'Resuming run with name {training_kws["run"]}, ID {run_id}!')
                # Or this is a new run
                else:
                    logger.debug(f'Starting new run with name {training_kws["run"]}!')
                # Start the run!
                with mlflow.start_run(run_name=training_kws["run"], run_id=run_id):
                    # If this is a new run, append the newly-created run ID to our yaml config file (if we passed this)
                    if args['config'] is not None and 'run_id' not in training_kws['mlflow_cfg'].keys():
                        new_run_id = mlflow.active_run().info.run_id
                        add_run_id_to_config_yaml(args["config"], new_run_id)
                        logger.debug(f'Added run id {new_run_id} to {args["config"]}!')
                    tm.start()

    # Running training locally
    else:
        tm = TrainingModule(**training_kws)
        tm.start()
