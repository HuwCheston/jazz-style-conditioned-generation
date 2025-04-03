#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Experiment with using CLaMP3-PPO for reinforcement learning"""

import heapq
import os
import random
from copy import deepcopy

import requests
import torch
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig

from clamp3.code.config import *
from clamp3.code.utils import CLaMP3Model, M3Patchilizer
from clamp3.preprocessing.midi.batch_midi2mtf import load_midi as clamp_load_midi
from jazz_style_conditioned_generation import utils, training
from jazz_style_conditioned_generation.data.dataloader import DatasetMIDIConditionedFullTrack, create_padding_mask

# Config file for the generator
GENERATIVE_MODEL_CFG = (
    "reinforcement-clamp-ppo/music_transformer_rpr_tsd_nobpe_conditionsmall_augment_schedule_10l8h_clampppo_2e5.yaml"
)
# Clamp3 checkpoints
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
        # This initialises our dataloaders, generative model, loads checkpoints, etc.
        super().__init__(**training_kwargs)

        # Reinitialise the optimizer from scratch
        self.optimizer_clamp = torch.optim.AdamW(self.model.parameters(), **self.optimizer_cfg["optimizer_kws"])

        # Make a deepcopy of the trained transformer, this is our reference model
        self.model_ref = deepcopy(self.model)

        # Raise an error if we haven't `git pull`ed CLaMP3
        if not os.path.isdir(os.path.join(utils.get_project_root(), "clamp3")):
            raise FileNotFoundError("Cannot find `clamp3` directory inside project root. "
                                    "Run `git clone https://github.com/sanderwood/clamp3.git` from root directory.")

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
        self.CLAMP3_PATCHILIZER = M3Patchilizer()

        # Load the checkpoint
        checkpoint = torch.load(CLAMP3_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        logger.info(
            f"Successfully Loaded CLaMP 3 Checkpoint from Epoch {checkpoint['epoch']} "
            f"with loss {checkpoint['min_eval_loss']}"
        )
        self.CLAMP3.load_state_dict(checkpoint['model'])

        self.worst_heap, self.best_heap = [], []  # Heaps to keep track of the top and bottom 10% of generations
        self.n_generations = 5  # number of generations to make per ground truth
        self.generation_keep = 0.2  # fraction of best and worst generations to keep
        self.generations_completed = 0  # number of generatios we've completed so far

        self.beta_ = 0.1
        self.lambda_ = 10

    def create_dataloaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """We only want to create a single full-track dataloader"""
        # Create training dataset loader: uses FULL tracks!
        train_loader = DataLoader(
            ClampGenerationLoader(
                tokenizer=self.tokenizer,
                files_paths=self.track_splits["train"][:10],
                max_seq_len=utils.MAX_SEQUENCE_LENGTH,
                **self.test_dataset_cfg
            ),
            batch_size=1,  # have to use a batch size of one for this class
            shuffle=False,  # don't want to shuffle either for this one
            drop_last=False,
            collate_fn=lambda x: x[0]  # just gives us a dictionary of one item
        )
        return train_loader, None, None

    def tokens_to_clamp(self, tokseq: torch.Tensor) -> torch.Tensor:
        """Converts a midi (either filename or symusic.Score) object to the format required for clamp"""
        # Convert the tokens into a score with the desired sampling rate
        midi = self.tokenizer.decode(tokseq).resample(utils.TICKS_PER_QUARTER)
        # Dump then load with the specialist CLaMP function
        midi.dump_midi("tmp.mid")
        clamp_midi = clamp_load_midi("tmp.mid", m3_compatible=True)
        # Encode with the patchilizer
        clamp_patches = torch.tensor(self.CLAMP3_PATCHILIZER.encode(clamp_midi, add_special_patches=True))
        # Cleanup
        os.remove("tmp.mid")
        return clamp_patches

    def extract_feature(self, patches: torch.Tensor) -> torch.Tensor:
        """Extracts features using CLaMP3. Ported from clamp3.code.extract_clamp3.extract_feature"""
        segment_list = []
        for i in range(0, len(patches), PATCH_LENGTH):
            segment_list.append(patches[i:i + PATCH_LENGTH])
        segment_list[-1] = patches[-PATCH_LENGTH:]

        # This code just copies what we get when we pass `filename.endswith(".mtf")` into extract_feature
        last_hidden_states_list = []
        for input_segment in segment_list:
            input_masks = torch.tensor([1] * input_segment.size(0))
            pad_indices = torch.ones(
                (PATCH_LENGTH - input_segment.size(0), PATCH_SIZE)
            ).long() * self.CLAMP3_PATCHILIZER.pad_token_id
            input_masks = torch.cat((input_masks, torch.zeros(PATCH_LENGTH - input_segment.size(0))), 0)
            input_segment = torch.cat((input_segment, pad_indices), 0)
            # Through the model
            with torch.zero_grad():
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
        top_size = max(1, int(self.generations_completed * self.generation_keep))  # Ensure at least 1 element is kept
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
        # Generate with just the conditioning tokens
        gen_i = self.model.generate(condition_tokens, target_seq_length=256 - condition_tokens.size(0))
        # Extract features from the generated track
        gen_i_clamp_data = self.tokens_to_clamp(gen_i)
        gen_i_clamp_features = self.extract_feature(gen_i_clamp_data)
        return gen_i, gen_i_clamp_features

    def step(self, batch):
        pass

    def training(self, epoch_num: int):
        pass

    def compute_log_probs(self, model, tokseq: torch.Tensor, no_grad: bool = False) -> torch.Tensor:
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
        # Sum everything together: shape (batch)
        return log_probs_sub.sum()

    def start(self):
        batch = next(iter(self.train_loader))
        # Evaluation mode for now
        self.model.eval()
        self.CLAMP3.eval()
        # Reset number of generations completed and heaps
        self.generations_completed = 0
        self.best_heap = []
        self.worst_heap = []
        # Extract features from the ground truth track
        track_clamp_data = self.tokens_to_clamp(batch["input_ids"])
        track_clamp_features = self.extract_feature(track_clamp_data)

        # Make N generations from the condition tokens
        for _ in tqdm(range(self.n_generations)):
            # Do the generation and extract features with clamp
            gen_i, gen_i_features = self.do_generation(batch["condition_ids"])
            # Compute the cosine similarity between generated and reference track
            gen_i_sim = torch.nn.functional.cosine_similarity(track_clamp_features, gen_i_features, dim=-1).item()
            self.generations_completed += 1
            # Add the generation and similarity scores to the heap if required
            self.add_to_heap(gen_i, gen_i_sim)

        logger.debug(f"Finished generating: {len(self.best_heap)} best items, {len(self.worst_heap)} worst items, "
                     f"best similarity {max(self.best_heap, key=lambda x: x[0])[0]:.3f}, "
                     f"worst similarity {max(self.worst_heap, key=lambda x: x[0])[0]:.3f}")
        # Shuffle both the best and worst generations
        random.shuffle(self.best_heap)
        random.shuffle(self.worst_heap)
        # Zip together (best generation, worst_generation), (best_generation, worst_generation)
        zipped = [(t1, t2) for (_, t1), (_, t2) in zip(self.best_heap, self.worst_heap)]

        # Now we're training
        self.model.train()
        for best, worst in tqdm(zipped):
            # # Compute log probabilities with the policy model
            # best_log_ps_policy_sum = self.compute_log_probs(self.model, best)
            # worst_log_ps_policy_sum = self.compute_log_probs(self.model, worst)
            # # Compute log probabilities with the reference model
            # best_log_ps_ref_sum = self.compute_log_probs(self.model_ref, best, no_grad=True)
            # worst_log_ps_ref_sum = self.compute_log_probs(self.model_ref, worst, no_grad=True)

            # Autoregressive label shifting
            best_input_ids, best_labels = best[:, :-1], best[:, :1]
            worst_input_ids, worst_labels = worst[:, :-1], worst[:, :1]
            # Create masks for padding tokens
            best_mask = create_padding_mask(best_input_ids, self.tokenizer.pad_token_id)
            worst_mask = create_padding_mask(worst_input_ids, self.tokenizer.pad_token_id)
            # Through the model to get logits for the policy model
            best_logits_policy = self.model(best_input_ids, best_labels, best_mask)
            worst_logits_policy = self.model(worst_input_ids, worst_labels, worst_mask)
            # Compute log probabilities
            best_log_ps_policy = torch.nn.functional.log_softmax(best_logits_policy, dim=-1)  # (batch, seq, vocab)
            worst_log_ps_policy = torch.nn.functional.log_softmax(worst_logits_policy, dim=-1)  # (batch, seq, vocab)
            # Subset to get only target token
            best_log_ps_policy_sub = torch.gather(
                best_log_ps_policy,
                dim=-1,
                index=best_labels.unsqueeze(-1)
            ).squeeze(-1)  # (batch, seq)
            worst_log_ps_policy_sub = torch.gather(
                worst_log_ps_policy,
                dim=-1,
                index=worst_labels.unsqueeze(-1)
            ).squeeze(-1)  # (batch, seq)
            # Sum everything
            best_log_ps_policy_sum = best_log_ps_policy_sub.sum()
            worst_log_ps_policy_sum = worst_log_ps_policy_sub.sum()
            # Clone everything
            best_input_ids_ref, best_labels_ref = best_input_ids.clone(), best_labels.clone()
            worst_input_ids_ref, worst_labels_ref = worst_input_ids.clone(), worst_labels.clone()
            best_mask_ref = best_mask.clone()
            worst_mask_ref = worst_mask.clone()
            # Same as above, but for the reference model: no need for us to keep track of computation graph here
            with torch.no_grad():
                best_logits_ref = self.model_ref(best_input_ids_ref, best_labels_ref, best_mask_ref).detach()
                worst_logits_ref = self.model_ref(worst_input_ids_ref, worst_labels_ref, worst_mask_ref).detach()
                # Log probabilities
                best_log_ps_ref = torch.nn.functional.log_softmax(best_logits_ref, dim=-1)  # (batch, seq, vocab)
                worst_log_ps_ref = torch.nn.functional.log_softmax(worst_logits_ref, dim=-1)
                # Subset for target tokens
                best_log_ps_ref_sub = torch.gather(
                    best_log_ps_ref,
                    dim=-1,
                    index=best_labels_ref.unsqueeze(-1)
                ).squeeze(-1)  # (batch, seq)
                worst_log_ps_ref_sub = torch.gather(
                    worst_log_ps_ref,
                    dim=-1,
                    index=worst_labels_ref.unsqueeze(-1)
                ).squeeze(-1)  # (batch, seq)
                # Sum everything
                best_log_ps_ref_sum = best_log_ps_ref_sub.sum()
                worst_log_ps_ref_sum = worst_log_ps_ref_sub.sum()

            # Loss computation
            logits = (best_log_ps_policy_sum - worst_log_ps_policy_sum) - (best_log_ps_ref_sum - worst_log_ps_ref_sum)
            loss = -torch.nn.functional.logsigmoid(
                self.beta_ * (logits - self.lambda_ * max(0, best_log_ps_ref_sum - best_log_ps_policy_sum))
            )
            logger.info(f"Loss: {loss.item():.3f}")
            # Backwards pass
            self.optimizer_clamp.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer_clamp.step()


if __name__ == "__main__":
    import argparse

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
    rm = ClampReinforcerModule(**cfg)
    rm.start()
