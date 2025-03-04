#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Sanity check simple training that just uses default settings for everything"""

from pathlib import Path
from random import shuffle

import numpy as np
import torch
from miditok import TokenizerConfig, REMI
from miditok.pytorch_data import DatasetMIDI, DataCollator
from miditok.utils import split_files_for_training
from tqdm import tqdm
from transformers import GPT2Config, GPT2LMHeadModel

from jazz_style_conditioned_generation import utils


def forwards_pass(batch, model) -> torch.tensor:
    input_ids = batch["input_ids"].to(utils.DEVICE)
    labels = batch["labels"].to(utils.DEVICE)
    attention_mask = batch["attention_mask"].to(utils.DEVICE)
    outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
    return outputs.loss


def main():
    # Grab all MIDI files
    midi_paths = [Path(p) for p in utils.get_data_files_with_ext("data/raw", "**/*.mid")]

    # Train REMI tokenizer using all default settings
    tokenizer_cfg = TokenizerConfig()
    tokenizer = REMI(tokenizer_cfg)
    tokenizer.train(files_paths=midi_paths, vocab_size=20000)

    # Create dummy data splits
    total_num_files = len(midi_paths)
    num_files_valid = round(total_num_files * 0.1)
    num_files_test = round(total_num_files * 0.1)
    shuffle(midi_paths)
    midi_paths_valid = midi_paths[:num_files_valid]
    midi_paths_test = midi_paths[num_files_valid:num_files_valid + num_files_test]
    midi_paths_train = midi_paths[num_files_valid + num_files_test:]

    # Create model
    model_cfg = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=1024,
        bos_token_id=tokenizer["BOS_None"],
        eos_token_id=tokenizer["EOS_None"],
    )
    model = GPT2LMHeadModel(model_cfg).to(utils.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    # Iterate through each subset
    for files_paths, subset_name in (
            (midi_paths_train, "train"), (midi_paths_valid, "valid"), (midi_paths_test, "test")
    ):
        # Split the MIDIs into chunks of sizes approximately about 1024 tokens
        subset_chunks_dir = Path(f"dataset_{subset_name}")
        split_paths = split_files_for_training(
            files_paths=files_paths,
            tokenizer=tokenizer,
            save_dir=subset_chunks_dir,
            max_seq_len=1024,
            num_overlap_bars=2,
        )
        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            DatasetMIDI(
                split_paths,
                tokenizer=tokenizer,
                max_seq_len=1024
            ),
            batch_size=4,
            collate_fn=DataCollator(
                pad_token_id=tokenizer.pad_token_id,
                copy_inputs_as_labels=True,
                shift_labels=True
            )
        )
        # Set model into the correct mode
        if subset_name == "train":
            model.train()
        else:
            model.eval()
        # Iterate through for one epoch
        epoch_loss = []
        for batch in tqdm(dataloader, desc=f"{subset_name.title()}ing:"):
            # Backwards pass for training only
            if subset_name == "train":
                loss = forwards_pass(batch, model)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    loss = forwards_pass(batch, model)
            epoch_loss.append(loss.item())
        print(f'Stage {subset_name}, loss: {np.mean(epoch_loss):.3f}')


if __name__ == "__main__":
    utils.seed_everything(utils.SEED)
    main()
