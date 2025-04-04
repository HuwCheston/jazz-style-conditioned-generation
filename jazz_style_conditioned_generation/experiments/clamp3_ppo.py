#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Experiment with using CLaMP3-PPO for reinforcement learning"""

import heapq
import os
import random
import sys
from copy import deepcopy
from time import time

import mlflow
import numpy as np
import requests
import torch
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig

from jazz_style_conditioned_generation import utils, training
from jazz_style_conditioned_generation.data.dataloader import DatasetMIDIConditionedFullTrack, create_padding_mask

sys.path.insert(0, os.path.join(utils.get_project_root()))

from clamp3.code.config import *
from clamp3.code.utils import CLaMP3Model, M3Patchilizer
from clamp3.preprocessing.midi.batch_midi2mtf import load_midi as clamp_load_midi


# Config file for the generator
GENERATIVE_MODEL_CFG = (
    "reinforcement-clamp-ppo/music_transformer_rpr_tsd_nobpe_conditionsmall_augment_schedule_10l8h_clampppo_2e6_TEST.yaml"
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

# Clamp3 model and configuration setup
audio_config = BertConfig(
    vocab_size=1,
    hidden_size=AUDIO_HIDDEN_SIZE,
    num_hidden_layers=AUDIO_NUM_LAYERS,
    num_attention_heads=AUDIO_HIDDEN_SIZE // 64,
    intermediate_size=AUDIO_HIDDEN_SIZE * 4,
    max_position_embeddings=MAX_AUDIO_LENGTH
)
symbolic_config = BertConfig(
    vocab_size=1,
    hidden_size=M3_HIDDEN_SIZE,
    num_hidden_layers=PATCH_NUM_LAYERS,
    num_attention_heads=M3_HIDDEN_SIZE // 64,
    intermediate_size=M3_HIDDEN_SIZE * 4,
    max_position_embeddings=PATCH_LENGTH
)


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


class ClampGenerationLoader(DatasetMIDIConditionedFullTrack):
    """Returns tokenized full tracks + condition tokens for generations"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, idx: int):
        loaded = deepcopy(self.preloaded_data[idx])
        full_score, _, metadata = loaded
        tokseq_ids = self.score_to_token_sequence(full_score, add_bos_eos=True)
        condition_tokens = self.get_conditioning_tokens(metadata)
        tokseq_ids = condition_tokens + tokseq_ids  # type: list[int]
        # TODO: consider passing input_ids directly through CLAMP here?
        # Return everything nicely formatted as a dictionary
        return {
            "input_ids": torch.tensor([tokseq_ids], dtype=torch.long),  # (batch, seq)
            "condition_ids": torch.tensor(condition_tokens, dtype=torch.long),  # (seq)
        }


class ClampReinforcerModule(training.FineTuningModule):
    def __init__(self, **training_kwargs):
        # Need to grab this and remove from kwargs before passing to the training module
        self.reinforce_cfg = training_kwargs.pop("reinforce_cfg", dict())

        # This initialises our dataloaders, generative model, loads checkpoints, etc.
        super().__init__(**training_kwargs)

        logger.info("----REINFORCEMENT LEARNING WITH CLAMP3----")

        # Make a deepcopy of the trained transformer, this is our reference model
        self.model_ref = deepcopy(self.model)
        logger.debug(f"Initialising reference model with {utils.total_parameters(self.model_ref)} parameters...")

        # Reinitialise the optimizer from scratch
        optimizer_type = self.optimizer_cfg.get("optimizer_type", "adam")
        optimizer_kws = self.optimizer_cfg.get("optimizer_kws", dict(lr=0.0001))
        logger.debug(f'Initialising clamp optimiser {optimizer_type} with parameters {optimizer_kws}...')
        betas = tuple(optimizer_kws.pop("betas", (0.9, 0.999)))
        self.optimizer_clamp = self.get_optimizer(optimizer_type)(self.model.parameters(), betas=betas, **optimizer_kws)

        # Download the checkpoint if we haven't done this already
        if not os.path.exists(CLAMP3_CHECKPOINT_PATH):
            logger.warning("CLaMP 3 checkpoints not found, downloading...")
            download_clamp3_checkpoints()

        # Initialise CLaMP3
        self.CLAMP3 = CLaMP3Model(
            audio_config=audio_config,
            symbolic_config=symbolic_config,
            text_model_name=TEXT_MODEL_NAME,
            hidden_size=CLAMP3_HIDDEN_SIZE,
            load_m3=CLAMP3_LOAD_M3
        ).to(utils.DEVICE)
        self.CLAMP3.eval()  # never train the feature extractor
        self.CLAMP3_PATCHILIZER = M3Patchilizer()

        # Load the checkpoint
        checkpoint = torch.load(CLAMP3_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        logger.debug(
            f"Successfully Loaded CLaMP 3 Checkpoint from Epoch {checkpoint['epoch']} "
            f"with loss {checkpoint['min_eval_loss']}"
        )
        self.CLAMP3.load_state_dict(checkpoint['model'])

        # Set parameters for reinforcement learning
        self.generated_sequence_length = self.reinforce_cfg.get("generated_sequence_length", utils.MAX_SEQUENCE_LENGTH)
        self.worst_heap, self.best_heap = [], []  # Heaps to keep track of the top and bottom 10% of generations
        self.all_similarities = []  # keep track of all similarity scores
        self.n_generations = self.reinforce_cfg.get("n_generations", 1000)  # number of generations to make per track
        self.generation_keep_proportion = self.reinforce_cfg.get("generation_keep_proportion", .1)  # % best/worst gens
        self.generations_completed = 0  # number of generations we've completed so far
        self.beta_ = self.reinforce_cfg.get("beta", .1)  # same as notagen
        self.lambda_ = self.reinforce_cfg.get("lambda", 10)  # same as notagen

        logger.debug(f"For each of our {len(self.train_loader)} training tracks, "
                     f"we'll generate {self.n_generations} tracks of {self.generated_sequence_length} tokens "
                     f"and keep the top/bottom {self.generation_keep_proportion * 100:.0f}%. ")
        logger.debug(f"We'll compute the loss with beta {self.beta_}, lambda {self.lambda_}.")

    def create_dataloaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """We only want to create a single full-track dataloader"""
        # Create training dataset loader: uses FULL tracks!
        train_loader = DataLoader(
            ClampGenerationLoader(
                tokenizer=self.tokenizer,
                files_paths=self.track_splits["train"],
                max_seq_len=utils.MAX_SEQUENCE_LENGTH,
                **self.test_dataset_cfg
            ),
            batch_size=1,  # have to use a batch size of one for this class
            shuffle=False,  # don't want to shuffle either for this one
            drop_last=False,
            collate_fn=lambda x: {k: v.to(utils.DEVICE) for k, v in x[0].items()}  # gives us a dictionary of one item
        )
        return train_loader, None, None

    def tokens_to_clamp(self, tokseq: torch.Tensor) -> torch.Tensor:
        """Converts a midi (either filename or symusic.Score) object to the format required for clamp"""
        # Convert the tokens into a score with the desired sampling rate
        midi = self.tokenizer.decode(tokseq.cpu()).resample(utils.TICKS_PER_QUARTER)
        # Dump then load with the specialist CLaMP function
        midi.dump_midi("tmp.mid")
        clamp_midi = clamp_load_midi("tmp.mid", m3_compatible=True)
        # Encode with the patchilizer
        encoded = self.CLAMP3_PATCHILIZER.encode(clamp_midi, add_special_patches=True)
        clamp_patches = torch.tensor(encoded).to(utils.DEVICE)
        # Cleanup
        os.remove("tmp.mid")
        return clamp_patches

    def extract_clamp_features(self, patches: torch.Tensor) -> torch.Tensor:
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
            ).long() * self.CLAMP3_PATCHILIZER.pad_token_id
            input_masks = torch.cat((
                input_masks,
                torch.zeros(PATCH_LENGTH - input_segment.size(0), device=utils.DEVICE)
            ), dim=0)
            input_segment = torch.cat((input_segment, pad_indices), 0)
            # Through the model
            with torch.no_grad():
                last_hidden_states = self.CLAMP3.get_symbolic_features(
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

    def add_to_heap(self, generation: torch.Tensor, similarity: float):
        """Update our best + worst heaps with tuples of (generated tokens, similarity with ground truth)"""
        # Ensure at least 1 element is always kept inside the heap
        top_size = max(1, int(self.generations_completed * self.generation_keep_proportion))
        # Maintain the best (largest scores) 10% elements
        if len(self.best_heap) < top_size:
            heapq.heappush(self.best_heap, (similarity, generation))  # Normal min-heap
        else:
            heapq.heappushpop(self.best_heap, (similarity, generation))
        # Maintain the worst (smallest scores) 10% elements
        if len(self.worst_heap) < top_size:
            heapq.heappush(self.worst_heap, (-similarity, generation))  # Store negative to simulate max-heap
        else:
            heapq.heappushpop(self.worst_heap, (-similarity, generation))

    def do_generation(self, condition_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Given a tensor of condition tokens, generate a track with `max_sequence_length`"""
        # Generate with just the conditioning tokens
        gen_i = self.model.generate(condition_tokens, target_seq_length=self.generated_sequence_length)
        # Extract features from the generated track
        gen_i_clamp_data = self.tokens_to_clamp(gen_i)
        gen_i_clamp_features = self.extract_clamp_features(gen_i_clamp_data)
        # Pad the generation to the desired length
        if gen_i.size(1) < self.generated_sequence_length:
            gen_i = torch.nn.functional.pad(
                gen_i,
                (0, self.generated_sequence_length - gen_i.size(1)),
                value=self.tokenizer.pad_token_id
            )
        # Return the generated token indices and the extracted clamp features
        return gen_i, gen_i_clamp_features

    def step_generation(self, batch: dict[str, torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Given a track, generate N tracks, compute similarity with ground truth, and return most/least similar"""
        # Extract features from the ground truth track: we use the entire track and then aggregate across chunks
        track_clamp_data = self.tokens_to_clamp(batch["input_ids"])
        track_clamp_features = self.extract_clamp_features(track_clamp_data)
        # Make N generations from the condition tokens
        for _ in tqdm(range(self.n_generations), desc="Generating..."):
            # Do the generation and extract features with clamp
            gen_i, gen_i_features = self.do_generation(batch["condition_ids"])
            # Compute the cosine similarity between generated and reference track
            gen_i_sim = torch.nn.functional.cosine_similarity(track_clamp_features, gen_i_features, dim=-1).item()
            self.all_similarities.append(gen_i_sim)  # keep track of all similarities scores for current ground truth
            self.generations_completed += 1
            # Add the generation and similarity scores to the heap if required
            self.add_to_heap(gen_i, gen_i_sim)
        # TODO: at this point, we should also discard tracks that have > 0.95 similarity to any training tracks
        #  as well as tracks that have other inherent problems (e.g., overt repetition?)
        # Remove the similarity score and return
        return [t for _, t in self.best_heap], [t for _, t in self.worst_heap]

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
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        # Subset to get probabilities for target token only: shape (batch, seq)
        log_probs_sub = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        # Zero out probabilities for mask tokens
        log_probs_sub *= ~mask
        # Sum everything together: shape (batch)
        return log_probs_sub.sum(dim=1)

    def step_loss(self, best: torch.Tensor, worst: torch.Tensor) -> torch.Tensor:
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
        loss = -torch.nn.functional.logsigmoid(
            self.beta_ * (logits - self.lambda_ * torch.nn.functional.relu(ref_pos_logps - policy_pos_logps))
        )
        # Average loss over all elements in the batch
        return loss.mean()

    def reset(self) -> None:
        """Resets everything for a new track"""
        self.CLAMP3.eval()
        self.model.eval()
        self.model_ref.eval()
        self.generations_completed = 0
        self.best_heap = []
        self.worst_heap = []
        self.all_similarities = []

    def reinforcement(self, ) -> tuple[float, float]:
        all_losses_mean, all_losses_sum = [], []
        # Each batch is a single track in the training dataset
        for batch_num, batch in enumerate(self.train_loader):
            self.reset()
            # Do generations and return lists of most and least similar to the ground truth track
            best_generations, worst_generations = self.step_generation(batch)
            # Compute summary statistics from all of our generations
            best_similarity = max(self.best_heap, key=lambda x: x[0])[0]
            worst_similarity = max(self.worst_heap, key=lambda x: x[0])[0]
            mean_similarity = np.mean(self.all_similarities)
            logger.debug(f"Finished generating for track {batch_num + 1}: "
                         f"generated {len(self.best_heap)} best items, {len(self.worst_heap)} worst items, "
                         f"best similarity {best_similarity:.3f}, "
                         f"worst similarity {worst_similarity:.3f}, "
                         f"average similarity {mean_similarity:.3f}")
            # Randomise the lists independently of each other
            random.shuffle(best_generations)
            random.shuffle(worst_generations)
            # Assemble the best and worst generations from this track into a tensor dataset
            temp_loader = DataLoader(
                torch.utils.data.TensorDataset(torch.cat(best_generations), torch.cat(worst_generations)),
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
            )
            # Now we're training
            self.model.train()
            total_loss = []
            # Iterate over randomised pairs of best and worst generations
            for best, worst in tqdm(temp_loader, desc="Computing loss..."):
                # Forwards
                loss = self.step_loss(best, worst)
                total_loss.append(loss.item())
                # Backwards
                self.optimizer_clamp.zero_grad()
                loss.backward()
                if self.clip_grad_norm > 0.:  # Clip gradients if required
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.optimizer_clamp.step()
            # Report metrics
            summed_loss, avg_loss = np.sum(total_loss), np.mean(total_loss)
            logger.debug(f"Finished computing loss for track {batch_num + 1}: "
                         f"summed loss is {summed_loss:.3f}, average is {avg_loss:.3f}")
            batch_metrics = dict(
                summed_reinforcement_loss=summed_loss,
                avg_reinforcement_loss=avg_loss,
                best_similarity=best_similarity,
                worst_similarity=worst_similarity,
                avg_similarity=mean_similarity
            )
            # Report results to MLFlow, if we're using this
            if self.mlflow_cfg.get("use", False):
                mlflow.log_metrics(batch_metrics, step=batch_num)
            # Append everything to the lists
            all_losses_mean.append(avg_loss)
            all_losses_sum.append(summed_loss)
        return np.mean(all_losses_mean), np.mean(all_losses_sum)

    def start(self):
        """Runs training for this module"""
        training_start = time()
        # Log parameters for the run to MLflow if required
        if self.mlflow_cfg.get("use", False):
            self.log_run_params_to_mlflow()
        # No epochs: this file just runs one iteration
        train_loss_mean, train_loss_sum = self.reinforcement()
        # Save the checkpoint
        iteration_metrics = dict(
            summed_reinforcement_loss_end=train_loss_sum,
            avg_reinforcement_loss_end=train_loss_mean,
            reinforcement_time=time() - training_start
        )
        checkpoint_path = os.path.join(self.checkpoint_dir, "reinforcement_iteration_1.pth")
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
    training.main(training_kws=cfg, trainer_cls=ClampReinforcerModule, config_fpath=parser_args["config"])
