#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Train a finetuned model with CLaMP3-DPO"""

import os
import random
import sys
from copy import deepcopy
from time import time
from typing import Union

import numpy as np
import requests
import torch
import torch.nn.functional as F
from loguru import logger
from miditok import MusicTokenizer
from symusic import Score
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
from transformers import BertConfig

from jazz_style_conditioned_generation import utils, training
from jazz_style_conditioned_generation.data.conditions import validate_condition_values, INCLUDE
from jazz_style_conditioned_generation.data.dataloader import create_padding_mask, DatasetMIDIConditionedNoOverlapChunks
from jazz_style_conditioned_generation.data.scores import load_score, preprocess_score

sys.path.insert(0, os.path.join(utils.get_project_root()))

from clamp3.code.config import *
from clamp3.code.utils import CLaMP3Model, M3Patchilizer
from clamp3.preprocessing.midi.batch_midi2mtf import load_midi as clamp_load_midi

# Config file for the generator
GENERATIVE_MODEL_CFG = (
    "reinforcement-clamp-ppo/"
    "music_transformer_rpr_tsd_nobpe_conditionsmall_augment_schedule_10l8h_clampppo_2e6_TEST.yaml"
)
# Clamp3 checkpoints: need to use the C2 version as this is optimised for symbolic music
CLAMP3_CHECKPOINT_NAME = (
    "weights_clamp3_c2_h_size_768_t_model_FacebookAI_xlm-roberta-base_t_length_128_a_size_768_"
    "a_layers_12_a_length_128_s_size_768_s_layers_12_p_size_64_p_length_512.pth"
)
CLAMP3_CHECKPOINT_PATH = os.path.join(
    utils.get_project_root(),
    "clamp3",
    CLAMP3_CHECKPOINT_NAME
)
CLAMP3_CHECKPOINT_URL = os.path.join(
    "https://huggingface.co/sander-wood/clamp3/resolve/main/",
    CLAMP3_CHECKPOINT_NAME
)
# Initialise CLaMP3: we'll load the checkpoint in our module
CLAMP3 = (
    CLaMP3Model(
        audio_config=BertConfig(
            vocab_size=1,
            hidden_size=AUDIO_HIDDEN_SIZE,
            num_hidden_layers=AUDIO_NUM_LAYERS,
            num_attention_heads=AUDIO_HIDDEN_SIZE // 64,
            intermediate_size=AUDIO_HIDDEN_SIZE * 4,
            max_position_embeddings=MAX_AUDIO_LENGTH
        ),
        symbolic_config=BertConfig(
            vocab_size=1,
            hidden_size=M3_HIDDEN_SIZE,
            num_hidden_layers=PATCH_NUM_LAYERS,
            num_attention_heads=M3_HIDDEN_SIZE // 64,
            intermediate_size=M3_HIDDEN_SIZE * 4,
            max_position_embeddings=PATCH_LENGTH
        ),
        text_model_name=TEXT_MODEL_NAME,
        hidden_size=CLAMP3_HIDDEN_SIZE,
        load_m3=CLAMP3_LOAD_M3
    )
    .to(utils.DEVICE)
    .eval()
)
CLAMP3_PATCHILIZER = M3Patchilizer()
EPS = 1e-3
MAX_COS_SIM = 0.95


def download_clamp3_checkpoints():
    """Downloads checkpoints for pretrained symbolic CLaMP3 from huggingface. Ported from clamp3.code.extract_clamp3"""
    response = requests.get(CLAMP3_CHECKPOINT_URL, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    with open(CLAMP3_CHECKPOINT_PATH, "wb") as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))


def midi_to_clamp(
        midi: Union[torch.Tensor, Score, str],
        tokenizer: MusicTokenizer = None
) -> torch.Tensor:
    """Converts a midi (either a sequence of tokens, Score, or filename) object to the format required for clamp"""
    # Input is a string: we can just load directly with the CLaMP function
    if isinstance(midi, str):
        clamp_midi = clamp_load_midi(midi, m3_compatible=True)
    # Input is a Score or token sequence: we need to dump the MIDI first and then load up with the specialist function
    else:
        if isinstance(midi, torch.Tensor):
            midi = tokenizer.decode(midi.cpu()).resample(utils.TICKS_PER_QUARTER)
        # Dump the midi then load with the specialist CLaMP function
        midi.dump_midi("tmp.mid")
        clamp_midi = clamp_load_midi("tmp.mid", m3_compatible=True)
    # Encode with the patchilizer and convert to a tensor
    encoded = CLAMP3_PATCHILIZER.encode(clamp_midi, add_special_patches=True)
    clamp_patches = torch.tensor(encoded).to(utils.DEVICE)
    # Cleanup
    if os.path.exists("tmp.mid"):
        os.remove("tmp.mid")
    return clamp_patches


def extract_clamp_features(patches: torch.Tensor) -> torch.Tensor:
    """Extracts features using CLaMP3. Ported from clamp3.code.extract_clamp3.extract_feature"""
    segment_list = []
    for i in range(0, len(patches), PATCH_LENGTH):
        segment_list.append(patches[i:i + PATCH_LENGTH])
    segment_list[-1] = patches[-PATCH_LENGTH:]

    # This code just copies what we get when we pass `filename.endswith(".mtf")` into extract_feature
    last_hidden_states_list = []
    for input_segment in segment_list:
        input_masks = torch.tensor([1] * input_segment.size(0), device=utils.DEVICE)
        pad_indices = torch.ones(
            (PATCH_LENGTH - input_segment.size(0), PATCH_SIZE), device=utils.DEVICE
        ).long() * CLAMP3_PATCHILIZER.pad_token_id
        input_masks = torch.cat((
            input_masks,
            torch.zeros(PATCH_LENGTH - input_segment.size(0), device=utils.DEVICE)
        ), dim=0)
        input_segment = torch.cat((input_segment, pad_indices), 0)
        # Through the model
        with torch.no_grad():
            last_hidden_states = CLAMP3.get_symbolic_features(
                symbolic_inputs=input_segment.unsqueeze(0).to(utils.DEVICE),
                symbolic_masks=input_masks.unsqueeze(0).to(utils.DEVICE),
                get_global=True
            )
        last_hidden_states_list.append(last_hidden_states)

    # We assume `get_global` = True here
    full_chunk_cnt = len(patches) // PATCH_LENGTH
    remain_chunk_len = len(patches) % PATCH_LENGTH
    if remain_chunk_len == 0:
        feature_weights = torch.tensor([PATCH_LENGTH] * full_chunk_cnt, device=utils.DEVICE).view(-1, 1)
    else:
        feature_weights = torch.tensor(
            [PATCH_LENGTH] * full_chunk_cnt + [remain_chunk_len], device=utils.DEVICE
        ).view(-1, 1)

    last_hidden_states_list = torch.concat(last_hidden_states_list, 0)
    last_hidden_states_list = last_hidden_states_list * feature_weights
    return last_hidden_states_list.sum(dim=0) / feature_weights.sum()


class GroundTruthDataset(Dataset):
    """Extracts features using CLaMP3 for all tracks associated with a given condition token"""

    def __init__(
            self,
            files_paths: list[str],
            condition_token: str,
    ):
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
        track_loaded = preprocess_score(load_score(fpath))
        # Convert the ground truth track into the format required for CLaMP
        gt_data = midi_to_clamp(track_loaded)
        # Extract features using CLaMP
        gt_features = extract_clamp_features(gt_data)
        return gt_features


class ReinforceTrainModule(training.FineTuningModule):
    """Module used to train a finetuned model with CLaMP3-DPO"""

    def __init__(self, **training_kwargs):
        # Need to grab this and remove from kwargs before passing to the training module
        self.reinforce_cfg = training_kwargs.pop("reinforce_cfg", dict())

        # This initialises our dataloaders, generative model, loads checkpoints, etc.
        super().__init__(**training_kwargs)

        logger.info("----REINFORCEMENT LEARNING WITH CLAMP3----")

        # These are the condition tokens we'll use in generation + evaluation
        self.condition_tokens = INCLUDE["genres"] + INCLUDE["pianist"]
        # These are the "music" tokens
        self.music_tokens = [
            i for i in self.tokenizer.vocab
            if not i.startswith(("PAD", "BOS", "EOS", "GENRE", "PIANIST", "TEMPO", "TIME", "RECORDING"))
        ]

        # Make a deepcopy of the trained transformer, this is our reference model
        self.model_ref = deepcopy(self.model)
        logger.debug(f"Initialising reference model with {utils.total_parameters(self.model_ref)} parameters...")

        # Reinitialise the optimizer from scratch
        optimizer_type = self.optimizer_cfg.get("optimizer_type", "adam")
        optimizer_kws = self.optimizer_cfg.get("optimizer_kws", dict(lr=0.0001))  # TODO: notagen used 1e-7/1e-6
        logger.debug(f'Initialising clamp optimiser {optimizer_type} with parameters {optimizer_kws}...')
        betas = tuple(optimizer_kws.pop("betas", (0.9, 0.999)))
        self.optimizer_clamp = self.get_optimizer(optimizer_type)(self.model.parameters(), betas=betas, **optimizer_kws)

        # Download the checkpoint if we haven't done this already
        if not os.path.exists(CLAMP3_CHECKPOINT_PATH):
            logger.warning("CLaMP 3 checkpoints not found, downloading...")
            download_clamp3_checkpoints()

        # Load the checkpoint for CLAMP
        checkpoint = torch.load(CLAMP3_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        logger.debug(
            f"Successfully Loaded CLaMP 3 Checkpoint from Epoch {checkpoint['epoch']} "
            f"with loss {checkpoint['min_eval_loss']}"
        )
        CLAMP3.load_state_dict(checkpoint['model'])

        # Set parameters for reinforcement learning
        self.generated_sequence_length = self.reinforce_cfg.get("generated_sequence_length", utils.MAX_SEQUENCE_LENGTH)
        self.n_generations = self.reinforce_cfg.get("n_generations", 1000)  # number of generations to use per track
        self.generation_keep_proportion = self.reinforce_cfg.get("generation_keep_proportion", .1)  # % best/worst gens

        # Values used in calculating the loss
        self.beta_ = self.reinforce_cfg.get("beta", .1)  # same as notagen
        self.lambda_ = self.reinforce_cfg.get("lambda", 10)  # same as notagen

        # Values used in ranking generations from worst -> best
        self.gamma_ = self.reinforce_cfg.get("gamma", .6)
        self.theta_ = self.reinforce_cfg.get("theta", .4)
        # When mean(abs(LL)) of a generation < thresh, we start penalising the model
        self.ll_threshold = self.reinforce_cfg.get("ll_threshold", 1.7)

        self.all_cosine_sims = []  # keep track of ALL generation cosine similarities
        self.all_loglikelihoods = []  # keep track of ALL generation repetition rates

        self.current_iteration = self.reinforce_cfg.get("current_iteration", 0)

        logger.debug(f"For each of our {len(self.train_loader)} training tracks, "
                     f"we'll generate {self.n_generations} tracks of {self.generated_sequence_length} tokens "
                     f"and keep the top/bottom {self.generation_keep_proportion * 100:.0f}%. ")
        logger.debug(f"We'll compute the loss with beta {self.beta_}, lambda {self.lambda_}.")

    def create_dataloaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Skip over creating training and validation dataloader, just create test"""
        # Create test dataset loader: uses FULL tracks, with no overlap between chunks
        # i.e., we go 0 - 100, 101 - 201, 202 - 302, etc., then average the loss over all chunks
        test_loader = DataLoader(
            DatasetMIDIConditionedNoOverlapChunks(
                tokenizer=self.tokenizer,
                files_paths=self.track_splits["test"],
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
    def generation_output_dir(self):
        fpath = os.path.join(
            utils.get_project_root(),
            "data/rl_generations",
            f"{self.experiment}/{self.run}"
        )
        if not os.path.isdir(fpath):
            os.makedirs(fpath)
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
            gen_i_midi = midi_to_clamp(midi_fpath)
            gen_i_toks = torch.load(pt_fpath, map_location=utils.DEVICE)
        # If we don't have the MIDI or there's a problem
        except (FileNotFoundError, IndexError) as e:
            raise FileNotFoundError(f"Could not load file at {midi_fpath}, raised error {e}")
        else:
            gen_i_features = extract_clamp_features(gen_i_midi)
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
        CLAMP3.eval()
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
            # Otherwise, average the cosine similarity across all ground truth tracks
            gen_i_cos_sim = cos_sim_all.mean().item()
            all_cos_sim.append(gen_i_cos_sim)
            # Compute the average log probability for the token sequence (ignoring pad tokens)
            gen_i_avg_ll = (self.compute_log_probs(self.model, gen_i_toks, no_grad=True) /
                            len(gen_i_toks[gen_i_toks != self.tokenizer.pad_token_id])).item()
            all_log_likelihood.append(gen_i_avg_ll)
            # Compute the score as (GAMMA * cos) - (THETA * max(0, LL_THRESH - |ll|)
            score = (self.gamma_ * gen_i_cos_sim) - (self.theta_ * max(0, self.ll_threshold - abs(gen_i_avg_ll)))
            all_res.append((gen_i_toks, score))
        # Extend the lists
        self.all_cosine_sims.extend(all_cos_sim)
        self.all_loglikelihoods.extend(all_log_likelihood)
        # Compute summary statistics from all of our generations
        logger.debug(f"Finished generating for token {token}: "
                     f"best cosine similarity {max(all_cos_sim):.3f}, "
                     f"worst cosine similarity {min(all_cos_sim):.3f}, "
                     f"average cosine similarity {np.mean(all_cos_sim):.3f}, "
                     f"best log-likelihood {max(all_log_likelihood):.3f}, "
                     f"worst log-likelihood {min(all_log_likelihood):.3f}, "
                     f"average log-likelihood {np.mean(all_log_likelihood):.3f}")
        # Get the best and worst generations
        best_gen, worst_gen = self.sort_generations(all_res)
        # Assemble the shuffled best and worst generations from this track into a tensor dataloader and return
        return DataLoader(
            TensorDataset(best_gen, worst_gen),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

    def step_loss(self, best: torch.Tensor, worst: torch.Tensor) -> torch.Tensor:
        self.model.train()
        # Forwards
        loss = self.compute_dpo_loss(best, worst)
        # Backwards
        self.optimizer_clamp.zero_grad()
        loss.backward()
        if self.clip_grad_norm > 0.:  # Clip gradients if required
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        self.optimizer_clamp.step()
        return loss.item()

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
        return np.mean(all_losses_mean), np.mean(all_losses_sum)

    def testing(self) -> tuple[float, float]:
        # Don't load a checkpoint up here, keep the model as it is following the reinforcement
        test_loss, test_accuracy = [], []
        # Iterate over every batch in the dataloader
        for batch_idx, batch in tqdm(
                enumerate(self.test_loader),
                total=len(self.test_loader),
                desc='Testing...'
        ):
            # Forwards pass
            with torch.no_grad():
                loss, accuracy = self.step(batch)
            # No backwards pass
            test_loss.append(loss.item())
            test_accuracy.append(accuracy.item())
        return np.mean(test_loss), np.mean(test_accuracy)

    def start(self):
        """Runs training for this module"""
        training_start = time()
        # Log parameters for the run to MLflow if required
        if self.mlflow_cfg.get("use", False):
            self.log_run_params_to_mlflow()
        # No epochs: this file just runs one iteration
        train_loss_mean, train_loss_sum = self.reinforcement()
        mean_cosine_sim, mean_log_likelihood = np.mean(self.all_cosine_sims), np.mean(self.all_loglikelihoods)
        logger.debug(f"Finished reinforcement iteration {self.current_iteration}: "
                     f"mean cosine similarity {mean_cosine_sim:.3f}, "
                     f"mean repetition rate {mean_log_likelihood:.3f}")
        # Do testing
        test_loss, test_acc = self.testing()
        logger.debug(f"Finished testing: loss {test_loss:.3f}, accuracy {test_acc:.3f}")
        # Save the checkpoint
        iteration_metrics = dict(
            summed_reinforcement_loss=train_loss_sum,
            mean_reinforcement_loss=train_loss_mean,
            mean_log_likelihood=mean_log_likelihood,
            mean_cosine_similarity=mean_cosine_sim,
            test_loss=test_loss,
            test_accuracy=test_acc,
            reinforcement_time=time() - training_start
        )
        checkpoint_path = os.path.join(self.checkpoint_dir, f"reinforcement_iteration_{self.current_iteration}.pth")
        self.save_checkpoint(iteration_metrics, checkpoint_path)


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
    # Run training
    training.main(training_kws=cfg, trainer_cls=ReinforceTrainModule, config_fpath=parser_args["config"])
