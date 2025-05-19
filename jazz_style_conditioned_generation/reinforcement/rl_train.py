#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Train a finetuned model with CLaMP3-DPO"""

import os
import random
from time import time

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from jazz_style_conditioned_generation import utils, training
from jazz_style_conditioned_generation.data.conditions import validate_condition_values, INCLUDE
from jazz_style_conditioned_generation.data.dataloader import create_padding_mask, DatasetMIDIConditionedNoOverlapChunks
from jazz_style_conditioned_generation.data.scores import load_score, preprocess_score
from jazz_style_conditioned_generation.reinforcement import clamp_utils

EPS = 1e-3
MAX_COS_SIM = 0.95
# Config file for the generator
GENERATIVE_MODEL_CFG = (
    "reinforcement-customtok-plateau/"
    "reinforcement_iter1_customtok_10msmin_lineartime_moreaugment_init6e5reduce10patience5_batch4_1024seq_12l8h768d3072ff.yaml"
)


class GroundTruthDataset(Dataset):
    """Extracts features using CLaMP3 for all tracks associated with a given condition token"""

    def __init__(
            self,
            files_paths: list[str],
            condition_token: str,
            clamp: clamp_utils.CLaMP3Model
    ):
        self.clamp = clamp
        self.condition_token = condition_token
        utils.validate_paths(files_paths, expected_extension=".mid")
        self.files_paths = list(self.get_tracks_with_condition(files_paths))

    def get_tracks_with_condition(self, files_paths: list[str]):
        """For every ground truth track with associated valid genres/pianist, extract features with CLaMP3"""
        for track in tqdm(
                files_paths,
                total=len(files_paths),
                desc=f"Getting ground truth tracks with condition {self.condition_token}"
        ):
            # Load in metadata for the track
            metadata = track.replace("piano_midi.mid", "metadata_tivo.json")
            metadata_read = utils.read_json_cached(metadata)
            # Get track pianist
            pianist = metadata_read["pianist"]
            validated_pianist = [pianist] if pianist in INCLUDE["pianist"] else []
            # Get genres and merge with other similar genres
            genres = [(n["name"], n["weight"]) for n in metadata_read["genres"]]
            validated_genres = validate_condition_values(genres, "genres")
            validated_genres = [i for i, _ in validated_genres]
            # Concatenate condition tokens together
            condition_tokens = validated_pianist + validated_genres
            # We only want to consider tracks with the desired condition token
            if self.condition_token in condition_tokens:
                yield track

    def __len__(self) -> int:
        return len(self.files_paths)

    def __getitem__(self, item: int) -> torch.Tensor:
        # Load up the scores and preprocess
        fpath = self.files_paths[item]
        track_tmp = load_score(fpath, as_seconds=True)
        track_loaded = preprocess_score(track_tmp)
        # Convert the ground truth track into the format required for CLaMP
        gt_data = clamp_utils.midi_to_clamp(track_loaded)
        # Extract features using CLaMP
        return clamp_utils.extract_clamp_features(gt_data, self.clamp)


class ReinforceTrainModule(training.TrainingModule):
    """Module used to train a finetuned model with CLaMP3-DPO"""

    def __init__(self, **training_kwargs):
        # Need to grab this and remove from kwargs before passing to the training module
        self.reinforce_cfg = training_kwargs.pop("reinforce_cfg", dict())
        self.n_test_tracks = self.reinforce_cfg.get("n_test_tracks", None)  # for faster debugging
        self.skip_training = self.reinforce_cfg.get("skip_training", False)  # only calculate ACS etc., don't train

        # This initialises our dataloaders, generative model, loads checkpoints, etc.
        super().__init__(**training_kwargs)
        logger.info("----REINFORCEMENT LEARNING WITH CLAMP3----")
        if self.skip_training:
            logger.warning("We will SKIP training and only calculate metrics (loss, ACS)")

        # These are the condition tokens we'll use in generation + evaluation
        self.condition_tokens = INCLUDE["genres"] + INCLUDE["pianist"]

        # Set parameters for reinforcement learning
        self.n_generations = self.reinforce_cfg.get("n_generations", 400)  # number of generations to use per track
        self.generation_keep_proportion = self.reinforce_cfg.get("generation_keep_proportion", .1)  # % best/worst gens

        # Values used in calculating the loss
        self.beta_ = self.reinforce_cfg.get("beta", .1)  # same as notagen
        self.lambda_ = self.reinforce_cfg.get("lambda", 10)  # same as notagen
        self.all_cosine_sims = []  # keep track of ALL generation cosine similarities
        self.current_iteration = self.reinforce_cfg.get("current_iteration", 0)
        logger.debug(f"We'll compute the loss with beta {self.beta_}, lambda {self.lambda_}.")

        self.test_cosine_sims = []  # keep track of cosine similarity of TEST (real) tracks to ground-truth tracks
        self.all_res = []  # List of dictionaries, converted into a JSON later on

        # Get model parameters
        model_type = self.model_cfg.get("model_type", "gpt2-lm")
        model_kws = self.model_cfg.get("model_kws", dict())
        logger.debug(f'Initialising policy + reference models {model_type} with arguments {model_kws}...')

        # Load the reference model, without loading scheduler + optimizer
        logger.debug(f"Loading reference model checkpoint from {self.reference_checkpoint_path}")
        self.model_ref = self.get_model(model_type, model_kws).to(utils.DEVICE)
        self.load_checkpoint(self.reference_checkpoint_path, weights_only=True, model=self.model_ref)

        # Load the policy model, without loading scheduler + optimizer
        logger.debug(f"Loading policy model checkpoint from {self.policy_checkpoint_path}")
        self.model = self.get_model(model_type, model_kws).to(utils.DEVICE)
        self.load_checkpoint(self.policy_checkpoint_path, weights_only=True, model=self.model)

        # Initialize the optimizer from scratch every iteration (same as NotaGen)
        self.initial_lr = self.optimizer_cfg["optimizer_kws"].get("lr", 0.0001)
        optimizer_type = self.optimizer_cfg.get("optimizer_type", "adam")
        optimizer_kws = self.optimizer_cfg.get("optimizer_kws", dict(lr=self.initial_lr))
        logger.debug(f'Initialising optimiser {optimizer_type} with parameters {optimizer_kws}...')
        betas = tuple(optimizer_kws.pop("betas", (0.9, 0.999)))
        self.optimizer = self.get_optimizer(optimizer_type)(self.model.parameters(), betas=betas, **optimizer_kws)

        # Initialize clamp3 from checkpoint
        self.clamp = clamp_utils.initialize_clamp(pretrained=True)

    @property
    def reference_checkpoint_path(self) -> str:
        """Path to the .pth file for the reference model"""
        fpath = self.reinforce_cfg.get("reference_model_checkpoint", None)
        # If the path doesn't exist, try adding the checkpoint directory to it
        if not os.path.exists(fpath):
            fpath = os.path.join(utils.get_project_root(), "checkpoints", fpath)
        # Validate that the filepath exists
        utils.validate_paths([fpath], expected_extension="pth")
        return fpath

    @property
    def policy_checkpoint_path(self):
        """Path to the .pth file for the policy model"""
        fpath = self.reinforce_cfg.get("policy_model_checkpoint", None)
        # If the path doesn't exist, try adding the checkpoint directory to it
        if not os.path.exists(fpath):
            fpath = os.path.join(utils.get_project_root(), "checkpoints", fpath)
        # Validate that the filepath exists
        utils.validate_paths([fpath], expected_extension="pth")
        return fpath

    def load_most_recent_checkpoint(self, weights_only: bool = True) -> None:
        """Override base method, we load checkpoints inside the __init__ now"""
        pass

    def create_dataloaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Skip over creating training and validation dataloader, just create test"""
        # Subset the number of test tracks if required
        if self.n_test_tracks is not None:
            test_tracks = self.track_splits["test"][:self.n_test_tracks]
        else:
            test_tracks = self.track_splits["test"]
        # Create test dataset loader: uses FULL tracks, with no overlap between chunks
        # i.e., we go 0 - 100, 101 - 201, 202 - 302, etc., then average the loss over all chunks
        test_loader = DataLoader(
            DatasetMIDIConditionedNoOverlapChunks(
                tokenizer=self.tokenizer,
                files_paths=test_tracks,
                max_seq_len=self.max_seq_len,
                **self.test_dataset_cfg  # most arguments can be shared across test + validation loader
            ),
            batch_size=self.batch_size,
            shuffle=False,  # don't want to shuffle for this one
            drop_last=False,
        )
        # Dummy training + validation loaders, real test loader
        return [], [], test_loader

    @property
    def generation_output_dir(self) -> str:
        """Gets the directory where MIDI generations are stored for this model"""
        # Get the filepath from the config
        if "generation_output_dir" in self.reinforce_cfg:
            fpath = os.path.join(
                utils.get_project_root(),
                "data/rl_generations",
                self.reinforce_cfg["generation_output_dir"]
            )
        # For backwards compatibility: get the filepath from the name of the finetuning experiment
        else:
            fpath = os.path.dirname(self.reference_checkpoint_path).replace("checkpoints", "data/rl_generations")
        if not os.path.isdir(fpath):
            raise FileNotFoundError(f"Couldn't find generations in {fpath}")
        return fpath

    def get_generation(self, token: str, generation_number: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Given a condition token and generation number, get the associated MIDI and return token + CLaMP features"""
        # Format the condition token into the form used by the tokenizer: I.e., `Hard-Bop` -> `GENRES_HardBop`
        # This is the filepath where we'll load/save the current MIDI
        fname = (f"{utils.remove_punctuation(token).replace(' ', '').lower()}_"
                 f"iter{str(self.current_iteration).zfill(3)}_"
                 f"gen{str(generation_number).zfill(3)}")
        midi_fpath = os.path.join(self.generation_output_dir, fname + ".mid")
        pt_fpath = os.path.join(self.generation_output_dir, fname + ".pt")
        # Try and load the MIDI and tensor from disk
        try:
            gen_i_midi = clamp_utils.midi_to_clamp(midi_fpath)
            gen_i_toks = torch.load(pt_fpath, map_location=utils.DEVICE)
        # If we don't have the MIDI or there's a problem
        except (FileNotFoundError, IndexError) as e:
            raise FileNotFoundError(f"Could not load file at {midi_fpath}, raised error {e}")
        else:
            gen_i_features = clamp_utils.extract_clamp_features(gen_i_midi, clamp=self.clamp)
            # Return the generated tokens and extracted CLaMP features
            return gen_i_toks, gen_i_features

    def compute_log_probs(self, model, tokseq: torch.Tensor, no_grad: bool = False) -> torch.Tensor:
        """Given a sequence of tokens, compute the log probabilities for the predicted token at every step and sum"""
        # Autoregressive label shifting
        tokseq_ = tokseq.clone()
        input_ids, labels = tokseq_[:, :-1], tokseq_[:, 1:]
        mask = create_padding_mask(input_ids, self.tokenizer.pad_token_id)
        # Through the desired model, with or without gradient computation: shape (batch, seq, vocab)
        if no_grad:
            with torch.no_grad():
                logits = model(input_ids, labels=labels, attention_mask=mask).detach()
        else:
            logits = model(input_ids, labels=labels, attention_mask=mask)
        # Compute log probabilities: still shape (batch, seq, vocab)
        log_probs = F.log_softmax(logits, dim=-1)
        # Subset to get probabilities for target token only: shape (batch, seq)
        log_probs_sub = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        # Zero out probabilities for mask tokens
        log_probs_sub *= ~mask
        # Sum everything together: shape (batch)
        return log_probs_sub.sum(dim=1)

    def compute_dpo_loss(self, best: torch.Tensor, worst: torch.Tensor) -> torch.Tensor:
        """Given a pair of best and worst generations, compute the DPO loss"""
        # Compute log probabilities with the policy model
        policy_pos_logps = self.compute_log_probs(self.model, best)
        policy_neg_logps = self.compute_log_probs(self.model, worst)
        # Compute log probabilities with the reference model
        ref_pos_logps = self.compute_log_probs(self.model_ref, best, no_grad=True)
        ref_neg_logps = self.compute_log_probs(self.model_ref, worst, no_grad=True)
        # Loss computation
        logits = (policy_pos_logps - policy_neg_logps) - (ref_pos_logps - ref_neg_logps)
        # Use relu(ref_+ - policy_+) rather than max(0, ref_+ - policy_+) for tensor compatibility
        # Loss will be within the range [0, inf] with shape (batch)
        loss = -F.logsigmoid(self.beta_ * (logits - self.lambda_ * F.relu(ref_pos_logps - policy_pos_logps)))
        # Average loss over all elements in the batch
        return loss.mean()

    def reset(self) -> None:
        """Resets everything for a new track"""
        self.clamp.eval()
        self.model.eval()
        self.model_ref.eval()

    def sort_generations(self, all_generations: list[tuple[torch.Tensor, float]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Given a list of [(generated tokens, score), (generated tokens, score)], sort into best + worst generations"""
        # Sort the list from lowest (worst) -> highest (best)
        all_res = sorted(all_generations, key=lambda x: x[1], reverse=False)
        # Remove the score values from the list, just keep the tokens
        all_gens = [ts for ts, _ in all_res]
        # Get the N best and worst generations
        n_generations = int(len(all_res) * self.generation_keep_proportion)
        worst_generations, best_generations = all_gens[:n_generations], all_gens[-n_generations:]
        # Randomise the lists independently of each other
        random.shuffle(best_generations)
        random.shuffle(worst_generations)
        # Return as tensors
        return torch.cat(best_generations), torch.cat(worst_generations)

    def step_generate(self, token: str, gt_features: torch.Tensor) -> DataLoader:
        """Scores generations and returns a dataloader of shuffled best + worst generations"""
        self.model.eval()
        # Start doing the generations
        all_cos_sim, all_log_likelihood = [], []
        all_res = []
        for gen_idx in tqdm(range(self.n_generations), desc="Scoring generations..."):
            # Pull the generated MIDI from disk and return the token sequence + CLaMP features
            gen_i_toks, gen_i_features = self.get_generation(token, gen_idx)
            # Compute the cosine similarity vs. every ground truth element
            cos_sim_all = F.cosine_similarity(gen_i_features, gt_features, dim=-1)
            # If we potentially have plagiarism (similarity > 0.95 for any track in ground truth), skip over
            if any(cos_sim_all > MAX_COS_SIM):
                continue
            # Otherwise, average the cosine similarity across all ground truth tracks to get the score
            score = cos_sim_all.mean().item()
            all_cos_sim.append(score)
            all_res.append((gen_i_toks, score))
        # Append results to list of dictionaries
        self.all_res.append(dict(token=token, cosine_sims=all_cos_sim, type="generated"))
        # Extend the lists of all our cosine similarities
        self.all_cosine_sims.extend(all_cos_sim)
        # Compute summary statistics from all of our generations
        logger.debug(f"Finished getting generations for token {token}: "
                     f"best cosine similarity {max(all_cos_sim):.3f}, "
                     f"worst cosine similarity {min(all_cos_sim):.3f}, "
                     f"average cosine similarity {np.mean(all_cos_sim):.3f}")
        # Get the best and worst generations
        best_gen, worst_gen = self.sort_generations(all_res)
        # Assemble the shuffled best and worst generations from this track into a tensor dataloader and return
        return DataLoader(
            TensorDataset(best_gen, worst_gen),
            batch_size=1,  # one best and worst pair == a batch
            shuffle=True,
            drop_last=False,
        )

    def step_loss(self, best: torch.Tensor, worst: torch.Tensor) -> torch.Tensor:
        # Small debug switch that we can use just to calculate ACS metrics, etc.
        if self.skip_training:
            with torch.no_grad():
                loss = self.compute_dpo_loss(best, worst)
        # Otherwise, actually do training
        else:
            self.model.train()
            # Forwards
            loss = self.compute_dpo_loss(best, worst)
            # Backwards
            self.optimizer.zero_grad()
            loss.backward()
            if self.clip_grad_norm > 0.:  # Clip gradients if required
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.optimizer.step()
        return loss.item()

    def test_train_similarity(self, gt_features: torch.Tensor, condition_token: str) -> float:
        """Computes cosine similarity between (real) test tracks and ground truth features"""
        # Define the dataloader with tracks from the TEST data that have this condition token
        gt_test_loader = DataLoader(
            GroundTruthDataset(
                self.track_splits["test"],
                condition_token=condition_token,
                clamp=self.clamp
            ),
            batch_size=1,
            shuffle=False,
            drop_last=False
        )
        test_sims = []
        # Iterate over all tracks in the test dataset with the corresponding condition token
        for test_features in gt_test_loader:
            # Compute the cosine similarity: dims (N_train_tracks)
            test_sim = F.cosine_similarity(test_features, gt_features, dim=-1)
            # Average to a scalar
            test_sims.append(test_sim.mean().item())
        # Append results to list of dictionaries
        self.all_res.append(dict(token=condition_token, cosine_sims=test_sims, type="real"))
        # Extend the list containing ALL of our test cosine similarities
        self.test_cosine_sims.extend(test_sims)
        # Return the mean for this condition token for logging
        return np.mean(test_sims)

    def reinforcement(self, ) -> tuple[float, float]:
        all_losses_mean, all_losses_sum = [], []
        # Iterate over all the desired condition tokens we want to use in generation
        for step, token in enumerate(self.condition_tokens):
            self.reset()
            # Define the dataloader used for this condition token
            gt_loader = DataLoader(
                GroundTruthDataset(
                    self.track_splits["train"],
                    condition_token=token,
                    clamp=self.clamp
                ),
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False
            )
            # Get all the features for all the ground truth tracks: shape (N_ground_truth, N_clamp_dims)
            gt_features = torch.cat([i for i in tqdm(gt_loader, desc="Extracting ground-truth features...")], dim=0)
            # Compute all of our generations, score versus the ground truth, and zip into (best, worst) pairs
            bw_loader = self.step_generate(token, gt_features)
            # Iterate over randomised pairs of best and worst generations and compute loss
            total_loss = [self.step_loss(b, w) for b, w in tqdm(bw_loader, f"Computing loss for token {token}...")]
            # Report metrics
            summed_loss, avg_loss = np.sum(total_loss), np.mean(total_loss)
            logger.debug(f"Finished for token {token}: "
                         f"summed loss is {summed_loss:.3f}, "
                         f"average is {avg_loss:.3f}")
            # Append everything to the lists
            all_losses_mean.append(avg_loss)
            all_losses_sum.append(summed_loss)
            # Compute the similarity between the ground truth tracks and the REAL tracks from the held-out test data
            real_sim = self.test_train_similarity(gt_features, token)
            logger.debug(f"Token {token}: test data similarity to ground-truth {real_sim:.3f}")
        return np.mean(all_losses_mean), np.mean(all_losses_sum)

    def testing(self) -> tuple[float, float, float, float]:
        # Don't load a checkpoint, keep the model as it is following the reinforcement iteration we've just done
        self.model.eval()
        test_loss_policy, test_accuracy_policy = [], []
        test_loss_ref, test_accuracy_ref = [], []
        # Iterate over every batch in the dataloader
        for batch in tqdm(self.test_loader, total=len(self.test_loader), desc='Testing...'):
            # Forwards pass with both policy and reference model
            with torch.no_grad():
                loss_policy, accuracy_policy = self.step(batch, model=self.model)
                loss_ref, accuracy_ref = self.step(batch, model=self.model_ref)
            # No backwards pass
            # Append policy model results
            test_loss_policy.append(loss_policy.item())
            test_accuracy_policy.append(accuracy_policy.item())
            # Append reference model results
            test_loss_ref.append(loss_ref.item())
            test_accuracy_ref.append(accuracy_ref.item())
        # Return average loss + accuracy for both policy and reference model
        return (np.mean(test_loss_policy), np.mean(test_accuracy_policy),
                np.mean(test_loss_ref), np.mean(test_accuracy_ref))

    def start(self):
        """Runs training for this module"""
        training_start = time()
        # No mlflow for reinforcement learning
        # No epochs: this file just runs one iteration
        train_loss_mean, train_loss_sum = self.reinforcement()
        mean_cosine_sim = np.mean(self.all_cosine_sims)
        logger.debug(f"Finished reinforcement iteration {self.current_iteration}: "
                     f"mean cosine similarity {mean_cosine_sim:.3f}")
        # Dump ACS metrics to a JSON
        js_path = os.path.join(self.checkpoint_dir, f"reinforcement_iteration_{self.current_iteration}.json")
        utils.write_json(self.all_res, js_path)
        # Do testing
        test_loss_pol, test_acc_pol, test_loss_ref, test_acc_ref = self.testing()
        test_sim_mean = np.mean(self.test_cosine_sims)
        logger.debug(f"Finished testing: policy loss {test_loss_pol:.3f}, accuracy {test_acc_pol:.3f}, "
                     f"reference loss {test_loss_ref:.3f}, accuracy {test_acc_ref:.3f}")
        logger.debug(f"Mean cosine similarity between test tracks and ground truth: {test_sim_mean:.3f}")
        # Save the checkpoint
        iteration_metrics = dict(
            summed_reinforcement_loss=train_loss_sum,
            mean_reinforcement_loss=train_loss_mean,
            mean_cosine_similarity=mean_cosine_sim,
            test_loss=test_loss_pol,
            test_accuracy=test_acc_pol,
            test_mean_cosine_similarity=test_sim_mean,
            reinforcement_time=time() - training_start
        )
        # Dump a checkpoint if we haven't been skipping training
        if not self.skip_training:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"reinforcement_iteration_{self.current_iteration}.pth")
            self.save_checkpoint(iteration_metrics, checkpoint_path)

    def save_checkpoint(self, epoch_metrics: dict, path: str) -> None:
        epoch_metrics["reinforcement"] = True  # add a flag to the checkpoint
        super().save_checkpoint(epoch_metrics, path)  # save the checkpoint as normal


if __name__ == "__main__":
    import argparse

    # Raise an error if we haven't `git pull`ed CLaMP3
    if not os.path.isdir(os.path.join(utils.get_project_root(), "clamp3")):
        raise FileNotFoundError("Cannot find `clamp3` directory inside project root directory. "
                                "Run `git clone https://github.com/sanderwood/clamp3.git` from root directory.")

    utils.seed_everything(utils.SEED)
    # Parsing arguments from the command line interface
    parser = argparse.ArgumentParser(description="Experiment with reinforcement learning for finetuning")
    parser.add_argument(
        "-c", "--config", default=GENERATIVE_MODEL_CFG, type=str,
        help="Path to config YAML file, relative to root folder of the project"
    )
    # Parse all arguments from the command line
    parser_args = vars(parser.parse_args())
    if not parser_args["config"]:
        raise ValueError("No config file specified")
    # Parse the config file
    cfg = training.parse_config_yaml(parser_args["config"])
    cfg["mlflow_cfg"]["use"] = False  # no mlflow!
    # Run training
    training.main(training_kws=cfg, trainer_cls=ReinforceTrainModule, config_fpath=parser_args["config"])
