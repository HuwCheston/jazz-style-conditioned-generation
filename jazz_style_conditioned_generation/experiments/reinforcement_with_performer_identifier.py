#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Experiment with using performer identification model for reinforcement learning"""

import os
import random
from time import time

import gymnasium as gym
import mlflow
import numpy as np
import torch
from loguru import logger
from symusic import Score
from torch.utils.data import DataLoader

from jazz_style_conditioned_generation import utils, training
from jazz_style_conditioned_generation.data.conditions import validate_condition_values
from jazz_style_conditioned_generation.data.dataloader import (
    DatasetMIDIConditionedRandomChunk,
    DatasetMIDIConditionedFullTrack
)
from jazz_style_conditioned_generation.encoders import load_performer_identifier, ResNet50, MusicTransformer

# Config file for the generator
GENERATIVE_MODEL_CFG = ("finetuning"
                        "/music_transformer_rpr_tsd_nobpe_conditionsmall_augment_schedule_10l8h_finetuning_2e5.yaml")
CLIP_LENGTH, FPS = 30, 100

PIANIST_MAPPING = {
    "Abdullah Ibrahim": 0,
    "Ahmad Jamal": 1,
    "Bill Evans": 2,
    "Brad Mehldau": 3,
    "Cedar Walton": 4,
    "Chick Corea": 5,
    # "Gene Harris": 6,    # no performer token
    # "Geri Allen": 7,    # no performer token
    "Hank Jones": 8,
    "John Hicks": 9,
    "Junior Mance": 10,
    "Keith Jarrett": 11,
    "Kenny Barron": 12,
    "Kenny Drew": 13,
    "McCoy Tyner": 14,
    "Oscar Peterson": 15,
    # "Stanley Cowell": 16,    # no performer token
    "Teddy Wilson": 17,
    "Thelonious Monk": 18,
    "Tommy Flanagan": 19,
}


class ReinforcementEnvironment(gym.Env):
    def __init__(
            self,
            generator: MusicTransformer,
            min_duration: int = 1.,  # TODO: change this
            alpha: float = 0.9
    ):
        self.tokenizer = generator.tokenizer
        self.generator = generator
        self.min_duration = min_duration  # amount of time sequence should last for before we calculate reward
        self.max_seq_len = 100  # TODO: change this

        # Grab conditioning tokens from tokenizer
        self.tempo_tokens = [i for i in self.tokenizer.vocab.keys() if i.startswith("TEMPOCUSTOM")]
        self.timesig_tokens = [i for i in self.tokenizer.vocab.keys() if i.startswith("TIMESIGNATURECUSTOM")]

        # Define the classifier: a pretrained performer identification module, used to compute reward
        self.classifier: ResNet50 = load_performer_identifier()

        # Define action space: each action is selecting one token
        self.action_space = gym.spaces.Discrete(self.tokenizer.vocab_size)
        # Define observation space: the sequence generated so far, with maximum length
        self.observation_space = gym.spaces.Box(
            low=min(self.tokenizer.vocab.values()),
            high=self.tokenizer.vocab_size,
            shape=(self.max_seq_len,),
            dtype=int
        )

        self.current_sequence = None
        self.current_class = None

        # Used for exponential moving average
        self.alpha = alpha
        self.running_avg = None

    def reset(self, seed: int = None, options: dict = None) -> torch.Tensor:
        super().reset(seed=seed)
        # Reset the running average for the current episode
        self.running_avg = None

        # Sample a random pianist
        pianist = random.choice(list(PIANIST_MAPPING.keys()))
        # Load up the metadata for this pianist
        pianist_fmt = pianist.replace(" ", "") + ".json"
        metadata_path = os.path.join(utils.get_project_root(), "references/tivo_artist_metadata", pianist_fmt)
        metadata_loaded = utils.read_json_cached(metadata_path)

        # Get the genres they are associated with and remove any duplicates, etc.
        genres = [(m["name"], m["weight"]) for m in metadata_loaded["genres"]]
        genres_validated = validate_condition_values(genres, "genres")

        # Make a weighted selection of associated genres with the assigned weights
        names, weights = zip(*genres_validated)
        genres_selected = utils.weighted_sample(names, weights, n_to_sample=5)
        genres_tokens = [f"GENRES_{utils.remove_punctuation(g).replace(' ', '')}" for g in genres_selected]

        # Format pianist token
        pianist_token = f"PIANIST_{utils.remove_punctuation(pianist).replace(' ', '')}"

        # Get a random tempo + time signature token
        tempo_token = random.choice(self.tempo_tokens)
        timesig_token = random.choice(self.timesig_tokens)

        # TODO: seed with music from jazznet here

        # Convert everything to token IDXs
        all_tokens = [pianist_token, *genres_tokens, tempo_token, timesig_token]
        token_idxs = [self.tokenizer[t] for t in all_tokens] + [self.tokenizer["BOS_None"]]

        # Set the current sequence + class to match the pianist
        self.current_sequence = torch.tensor([token_idxs], dtype=torch.long)  # (batch, seq)
        self.current_class = PIANIST_MAPPING[pianist]
        return self.current_sequence

    def step(self, action: torch.Tensor) -> tuple:
        # Add the predicted token to the next step of our sequence
        action = action.unsqueeze(0)  # (batch, 1)
        self.current_sequence = torch.cat([self.current_sequence, action], dim=-1)
        # Stop generating once we've reached the desired sequence length or the model finishes generation early
        if (
                self.current_sequence.size(1) >= self.max_seq_len
                or action.item() == self.tokenizer["EOS_None"]
        ):
            terminated = True
        else:
            terminated = False
        # Decode the sequence into a score with the tokenizer
        score = self.tokenizer.decode(self.current_sequence).resample(utils.TICKS_PER_QUARTER)  # 1 tick == 1 ms
        # No reward calculated until the sequence is long enough
        current_duration = score.end() / 1000  # end time expressed in seconds
        # TODO: think about whether we want to set this to 0. or discard in `finalise_episode`
        if current_duration < self.min_duration:
            reward = 0.
        # Sequence is long enough, we can compute the reward
        else:
            roll = self.score_to_piano_roll(score)  # (batch, 1, 88, 3000)
            reward = self.compute_reward(roll)
        return self.current_sequence, reward, terminated, False

    def compute_reward(self, piano_roll: torch.Tensor) -> torch.Tensor:
        # Through the model to get (1, n_classes)
        logits = self.classifier(piano_roll)
        # Convert to probabilities and get the probability for the class we used for our condition token
        smaxed = torch.nn.functional.softmax(logits, dim=-1)
        target_prob = smaxed[:, self.current_class].detach().item()
        # Compute the running average of all probabilities
        if self.running_avg is None:
            self.running_avg = target_prob  # Initialize with the first logit
        else:
            self.running_avg = (self.alpha * self.running_avg + (1 - self.alpha) * target_prob)  # EMA update
        # Compute the reward as current_probability - previous_probabilities
        #  Such that positive values == more likely to be desired performer, negative == less likely
        return target_prob - self.running_avg

    def score_to_piano_roll(self, score: Score) -> torch.Tensor:
        """Converts a symusic.Score object to a piano roll with shape (batch, channel, height, width)"""
        # Convert to a piano roll
        roll = score.tracks[0].pianoroll(
            modes=["frame"],  # we don't care about separate onset/offset rolls
            pitch_range=(utils.MIDI_OFFSET, utils.MIDI_OFFSET + utils.PIANO_KEYS),  # gives us desired height
            encode_velocity=True
        )
        downsampled = roll[:, :, ::10]  # downsamples from 1 column == 1 ms -> 1 column == 10 ms
        # Pads to shape (channel, 88, 3000), as used in performer identification model initially
        desired_width = CLIP_LENGTH * FPS
        if downsampled.shape[-1] < desired_width:
            clip = np.pad(
                downsampled,
                (
                    (0, 0),
                    (0, 0),
                    (0, desired_width - downsampled.shape[-1])
                ),
                mode="constant",
                constant_values=self.tokenizer.pad_token_id
            )
        # Truncate from end to get shape (channel, 88, 3000)
        else:
            clip = downsampled[:, :, -desired_width:]
        # Normalize to within the range (0, 1)
        normalized = (clip - np.min(clip)) / (np.max(clip) - np.min(clip))
        return (
            torch.tensor(normalized)
            .to(torch.float32)
            .to(utils.DEVICE)
            .unsqueeze(0)
        )


class ReinforcementRunner(training.FineTuningModule):
    def __init__(
            self,
            episodes: int = 100,
            alpha: float = 0.9,
            gamma: float = 0.99,
            training_kwargs: dict = None
    ):
        # Set a few parameters
        # TODO: disable these
        training_kwargs["mlflow_cfg"]["use"] = False  # no mlflow
        training_kwargs["checkpoint_cfg"]["checkpoint_dir"] = os.path.join(utils.get_project_root(), "checkpoints")
        training_kwargs["pretrained_checkpoint_path"] = os.path.join(
            utils.get_project_root(),
            "checkpoints/pretraining-tsd/"
            "music_transformer_rpr_tsd_nobpe_conditionsmall_augment_schedule_10l8h_pretraining_2e5/validation_best.pth"
        )

        # Initialise the training module, load checkpoints, get the tokenizer etc.
        super().__init__(**training_kwargs)

        # Initialise environment
        self.env = ReinforcementEnvironment(self.model, alpha=alpha)

        # Initialise metrics from scratch
        self.current_epoch = 0
        self.current_validation_loss = 0.
        self.best_validation_loss = 1e4
        self.best_training_reward = -1e10

        self.episodes = episodes  # number of episodes to perform before validating
        self.gamma = gamma  # discount factor

        self.rewards = []
        self.saved_log_probs = []

    def create_dataloaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
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
        # No training loader needed, we generate data inside our environment
        return None, validation_loader, test_loader

    def finish_episode(self) -> torch.Tensor:
        """REINFOCE loss function, from https://github.com/pytorch/examples"""
        R = 0

        returns = []
        # Iterate over all rewards in REVERSE order
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            # This ensures that we build up the list in the correct (non-reversed) order
            returns.insert(0, R)
        returns = torch.tensor(returns)
        # Normalize the returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)
        # Iterate over the log probability of the sampled step at every
        policy_loss = []
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        return torch.stack(policy_loss).sum()

    def reinforce_step(self, ) -> tuple[torch.Tensor, float]:
        # Compute the initial sequence
        state = self.env.reset()
        # Reset lists for this episode
        self.rewards = []
        self.saved_log_probs = []
        # Keep iterating until we hit the desired sequence length
        done = False
        ep_reward = 0.  # keeps track of total reward for this episode
        while not done:
            # Compute the next action
            action, log_probs = self.model.predict_next_token(state)
            # Compute the state with the current action and the reward function
            state, reward, done, _ = self.env.step(action)
            # Store the log probabilities and reward for this action
            self.saved_log_probs.append(log_probs)
            self.rewards.append(reward)
            ep_reward += reward
        # Compute the REINFORCE loss function
        loss = self.finish_episode()
        # Return the loss and the sum total of all rewards from this step
        return loss, ep_reward

    def training(self, epoch_num: int):
        self.model.train()
        all_rewards = []
        for _ in range(self.episodes):
            # Forwards pass
            loss, ep_reward = self.reinforce_step()  # no batch to pass in
            # Backwards pass
            self.optimizer.zero_grad()
            loss.backward()
            if self.clip_grad_norm > 0.:  # Clip gradients if required
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.optimizer.step()
        # Return average reward for an episode
        return np.mean(all_rewards)

    def start(self):
        training_start = time()
        for epoch in range(self.current_epoch, self.epochs):
            epoch_start = time()
            # Training with REINFORCE
            avg_episode_reward = self.training(epoch)
            logger.info(f'... epoch {epoch}, avg. reward {avg_episode_reward:.3f}')
            # Validation
            self.current_validation_loss, validation_accuracy = self.validation(epoch)
            logger.debug(f'Epoch {epoch} / {self.epochs}, validation finished: '
                         f'loss {self.current_validation_loss:.3f}, accuracy {validation_accuracy:.3f}')
            # Log if this is our best epoch
            if self.best_training_reward < avg_episode_reward:
                self.best_training_reward = avg_episode_reward
                logger.info(f"New best training reward: {self.best_training_reward:.3f}")
            if self.current_validation_loss < self.best_validation_loss:
                self.best_validation_loss = self.current_validation_loss
                logger.info(f'New best validation loss: {self.current_validation_loss:.3f}')
            # Log parameters from this epoch in MLFlow
            epoch_metrics = dict(
                epoch_time=time() - epoch_start,
                train_reward=avg_episode_reward,
                current_validation_loss=self.current_validation_loss,
                best_validation_loss=self.best_validation_loss,
                validation_accuracy=validation_accuracy,
                lr=self.get_scheduler_lr()
            )
            # Report results to MLFlow, if we're using this
            if self.mlflow_cfg.get("use", False):
                mlflow.log_metrics(epoch_metrics, step=epoch)
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


if __name__ == "__main__":
    utils.seed_everything(utils.SEED)

    cfg = training.parse_config_yaml(GENERATIVE_MODEL_CFG)
    rm = ReinforcementRunner(training_kwargs=cfg)
    rm.start()
