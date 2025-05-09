#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility functions and methods for working with CLAMP-3"""

import os
import sys
from typing import Union

import requests
import torch
from loguru import logger
from miditok import MusicTokenizer
from symusic import Score
from tqdm import tqdm
from transformers import BertConfig

from jazz_style_conditioned_generation import utils

sys.path.insert(0, os.path.join(utils.get_project_root()))

from clamp3.code.config import *
from clamp3.code.utils import CLaMP3Model, M3Patchilizer
from clamp3.preprocessing.midi.batch_midi2mtf import load_midi as clamp_load_midi

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


def extract_clamp_features(patches: torch.Tensor, clamp: CLaMP3Model) -> torch.Tensor:
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
            last_hidden_states = clamp.get_symbolic_features(
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


def initialize_clamp(pretrained: bool = True) -> CLaMP3Model:
    # Initialize the model
    clamp = (
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

    # Download the checkpoint if we haven't done this already
    if pretrained:
        if not os.path.exists(CLAMP3_CHECKPOINT_PATH):
            logger.warning("CLaMP 3 checkpoints not found, downloading...")
            download_clamp3_checkpoints()

        # Load the checkpoint for CLAMP
        checkpoint = torch.load(CLAMP3_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        logger.debug(
            f"Successfully Loaded CLaMP 3 Checkpoint from Epoch {checkpoint['epoch']} "
            f"with loss {checkpoint['min_eval_loss']}"
        )
        clamp.load_state_dict(checkpoint['model'])

    return clamp
