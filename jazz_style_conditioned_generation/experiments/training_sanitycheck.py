#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Sanity check simple training that just uses default settings for everything"""

import os
import random
from pathlib import Path

import numpy as np
import torch
from miditok import TokenizerConfig, Structured
from miditok.pytorch_data import DatasetMIDI, DataCollator
from miditok.utils import split_files_for_training
from tqdm import tqdm
from transformers import GPT2Config, GPT2LMHeadModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEQUENCE_LEN = 1024
N_FILES = 1000
N_EPOCHS_PER_VOCAB = 3

# We'll train a model with this vocab size for N_EPOCHS_PER_SETTING epochs
VOCAB_SIZES = [-1, 500, 1000, 5000, 10000]  # -1 vocab size == no training of tokenizer


def seed_everything(seed: int = 42) -> None:
    """Sets all random seeds for reproducible results."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # safe to call even if cuda is not available
    random.seed(seed)
    np.random.seed(seed)


def forwards_pass(batch, model, pad_token_id) -> tuple[torch.tensor, torch.tensor]:
    input_ids = batch["input_ids"].to(DEVICE)
    labels = batch["labels"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)  # boolean required for music transformer
    outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
    accuracy = accuracy_score(outputs.logits, labels, pad_token_id)
    return outputs.loss, accuracy


def accuracy_score(logits: torch.tensor, labels: torch.tensor, pad_token_id: int) -> torch.tensor:
    # For each step in the sequence, this is the predicted label
    predicted = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)
    # True if the label is not a padding token, False if it is a padding token
    non_padded: torch.tensor = labels != pad_token_id
    # Get the cases where the predicted label is the same as the actual label
    correct = (predicted == labels) & non_padded
    # Calculate the accuracy from this
    return correct.sum().item() / non_padded.sum().item()


def get_project_root() -> str:
    """Returns the root directory of the project"""
    # Possibly the root directory, but doesn't work when running from the CLI for some reason
    poss_path = str(Path(__file__).parent.parent)
    # The root directory should always have these files (this is pretty hacky)
    if all(fp in os.listdir(poss_path) for fp in ["config", "checkpoints", "data", "outputs", "setup.py"]):
        return poss_path
    else:
        return os.path.abspath(os.curdir)


def get_data_files_with_ext(dir_from_root: str = "data/raw", ext: str = "**/*.mid") -> list[str]:
    return [p for p in Path(os.path.join(get_project_root(), dir_from_root)).glob(ext)]


def main():
    # Grab N MIDI files
    midi_paths = get_data_files_with_ext("data/raw", "**/*.mid")[:N_FILES]

    # Create dummy data splits
    total_num_files = len(midi_paths)
    num_files_test = round(total_num_files * 0.2)  # 80-20 train-test split
    random.shuffle(midi_paths)
    midi_paths_test = midi_paths[:num_files_test]
    midi_paths_train = midi_paths[num_files_test:]
    print(f'N {len(midi_paths_train)} train, N {len(midi_paths_test)} test')

    for vocab in VOCAB_SIZES:
        # Train structured tokenizer using all default settings
        tokenizer_cfg = TokenizerConfig()
        tokenizer = Structured(tokenizer_cfg)
        # Train tokenizer only when required
        if vocab > tokenizer.vocab_size:
            tokenizer.train(files_paths=midi_paths, vocab_size=vocab)
        print(f'Using tokenizer: {tokenizer}')
        # Create model
        model_cfg = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            n_positions=MAX_SEQUENCE_LEN,
            bos_token_id=tokenizer["BOS_None"],
            eos_token_id=tokenizer["EOS_None"],
        )
        model = GPT2LMHeadModel(model_cfg).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

        # Iterate through each subset
        for files_paths, subset_name in (
                (midi_paths_train, "train"), (midi_paths_test, "test")
        ):
            # Split the MIDIs into chunks of sizes approximately about 1024 tokens
            subset_chunks_dir = Path(f"dataset_{subset_name}")
            split_paths = split_files_for_training(
                files_paths=files_paths,
                tokenizer=tokenizer,
                save_dir=subset_chunks_dir,
                max_seq_len=MAX_SEQUENCE_LEN,
                num_overlap_bars=2,
            )
            # Create dataloader
            dataloader = torch.utils.data.DataLoader(
                DatasetMIDI(
                    split_paths,
                    tokenizer=tokenizer,
                    max_seq_len=MAX_SEQUENCE_LEN
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
            # Iterate over for the required number of epochs
            for n_epoch in range(1, N_EPOCHS_PER_VOCAB + 1):
                epoch_loss, epoch_acc = [], []
                for batch in tqdm(dataloader, desc=f"{subset_name.title()}ing, vocab {vocab}, epoch {n_epoch}"):
                    # Backwards pass for training only
                    if subset_name == "train":
                        loss, accuracy = forwards_pass(batch, model, tokenizer["PAD_None"])
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    # No backwards pass for testing
                    else:
                        with torch.no_grad():
                            loss, accuracy = forwards_pass(batch, model, tokenizer["PAD_None"])
                    epoch_loss.append(loss.item())
                    epoch_acc.append(accuracy)
                # Log to the console
                print(f'Stage {subset_name}, vocab {vocab}, epoch {n_epoch} '
                      f'loss: {np.mean(epoch_loss):.3f}, accuracy: {np.mean(epoch_acc):.3f}')


if __name__ == "__main__":
    seed_everything()
    main()
