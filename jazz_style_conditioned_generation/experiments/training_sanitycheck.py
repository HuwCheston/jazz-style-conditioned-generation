#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Sanity check simple training that just uses default settings for everything"""

import json
import os
import random
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from miditok import TokenizerConfig, REMI, MIDILike, TSD, Structured, PerTok
from miditok.pytorch_data import DatasetMIDI, DataCollator
from miditok.utils import split_files_for_training
from torchmetrics.text import Perplexity
from tqdm import tqdm

from jazz_style_conditioned_generation import metrics
from jazz_style_conditioned_generation.encoders import MusicTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEQUENCE_LEN = 1024
N_FILES = 1000
N_EPOCHS_PER_VOCAB = 10
N_EXAMPLES_TO_GENERATE = 50

# We'll train a model with this vocab size for N_EPOCHS_PER_SETTING epochs
SKIP_VOCAB, SKIP_TOKENIZER = [], []
VOCAB_SIZES = [-1, 500, 750, 1000, 2500, 5000, 7500, 10000]  # -1 vocab size == no training of tokenizer
TOKENIZER_TYPES = ["midilike", "structured", "tsd", "pertok"]

TOKENIZER_CFG = {
    "pitch_range": (21, 109),
    # TODO:for safety, this should probably be {(0, 4): 32}
    "beat_res": {(0, 1): 128},
    "num_velocities": 32,
    "special_tokens": [
        "PAD",  # add for short inputs to ensure consistent sequence length for all inputs
        "BOS",  # beginning of sequence
        "EOS",  # end of sequence
        "MASK",  # prevent attention to future tokens
    ],
    "use_chords": False,
    "use_rests": False,
    "use_tempos": False,
    "use_time_signatures": False,
    "use_programs": False,
    "use_sustain_pedals": False,
    "use_pitch_bends": False,
    "use_velocities": True,
    "remove_duplicated_notes": True,
    "encode_ids_split": "no",
    "use_pitchdrum_tokens": False,
    "programs": [0],  # only piano
}
PERTOK_CFG = {  # only for use in pertok!
    "use_microtiming": True,
    "ticks_per_quarter": 384,
    "max_microtiming_shift": 0.125,
    "num_microtiming_bins": 30,
}


def get_tokenizer_class_from_string(tokenizer_type: str):
    """Given a string, return the correct tokenizer class"""
    valids = ["remi", "midilike", "tsd", "structured", "pertok"]
    tokenizer_type = tokenizer_type.lower()
    if tokenizer_type == "remi":
        return REMI
    elif tokenizer_type == "midilike":
        return MIDILike
    elif tokenizer_type == "tsd":
        return TSD
    elif tokenizer_type == "structured":
        return Structured
    elif tokenizer_type == "pertok":
        return PerTok
    else:
        raise ValueError(f'`tokenizer_type` must be one of {", ".join(valids)} but got {tokenizer_type}')


def seed_everything(seed: int = 42) -> None:
    """Sets all random seeds for reproducible results."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # safe to call even if cuda is not available
    random.seed(seed)
    np.random.seed(seed)


def forwards_pass(batch, model, pad_token_id) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
    input_ids = batch["input_ids"].to(DEVICE)
    labels = batch["labels"].to(DEVICE)
    labels[labels == -100] = pad_token_id  # data collator seems to somehow replace 0 (pad) with -100 for labels?
    attention_mask = (input_ids == pad_token_id).float().to(DEVICE)  # boolean required for music transformer
    outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
    ppl = Perplexity(ignore_index=pad_token_id).to(DEVICE)
    accuracy = accuracy_score(outputs.logits, labels, pad_token_id)
    return outputs.loss, accuracy, ppl(outputs.logits, labels)


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


def save_results(res: list[dict], tokeniser_type: str, vocab_size: str) -> None:
    out = os.path.join(get_project_root(), "outputs/sanity_check",
                       f"sanitycheck_results_{tokeniser_type}_{vocab_size}.json")
    with open(out, "w") as fp:
        json.dump(res, fp, indent=4, ensure_ascii=False, sort_keys=False)


def generate_examples(
        test_dataloader: torch.utils.data.DataLoader,
        model: MusicTransformer,
        primer_len: int = 128,
        n_generations: int = N_EXAMPLES_TO_GENERATE
) -> dict:
    gens = []
    with tqdm(total=n_generations, desc="Generating...") as pgbar:
        for example in test_dataloader:
            for primer in example["input_ids"]:
                no_pad = [i for i in primer if i != model.tokenizer.pad_token_id and i != -100]
                primer_trunc = torch.tensor(no_pad[:primer_len])
                gen = model.generate(primer_trunc, MAX_SEQUENCE_LEN)
                if len(gens) > n_generations:
                    break
                gens.append(gen.detach().to("cpu"))
                pgbar.update(1)
            if len(gens) > n_generations:
                break
    return gens


def main(tokeniser_type: str, vocab_sizes: list[int]):
    # Grab N MIDI files
    midi_paths = sorted(get_data_files_with_ext("data/raw", "**/*.mid"))[:N_FILES]

    # Create dummy data splits
    total_num_files = len(midi_paths)
    num_files_test = round(total_num_files * 0.2)  # 80-20 train-test split
    # random.shuffle(midi_paths)
    midi_paths_test = midi_paths[:num_files_test]
    midi_paths_train = midi_paths[num_files_test:]
    print(f'N {len(midi_paths_train)} train, N {len(midi_paths_test)} test')
    print(f'Tokenizer type {tokeniser_type}, configuration: {TOKENIZER_CFG}')
    results = []

    for vocab in vocab_sizes:
        if vocab in SKIP_VOCAB and tokeniser_type in SKIP_TOKENIZER:
            continue

        # Train structured tokenizer using all default settings
        tokcfg = TOKENIZER_CFG if tokeniser_type != "pertok" else TOKENIZER_CFG | PERTOK_CFG
        tokenizer_cfg = TokenizerConfig(**tokcfg)
        tokenizer = get_tokenizer_class_from_string(tokeniser_type)(tokenizer_cfg)
        # Train tokenizer only when required
        if vocab > tokenizer.vocab_size:
            tokenizer.train(files_paths=midi_paths, vocab_size=vocab)
        # Measure musical statistics from data using tokenizer
        mp = metrics.compute_metrics_for_dataset(midi_paths, tokenizer)
        for metric_name, metric_val in mp.items():
            print(f'{metric_name}: {metric_val}')
        results.append(dict(epoch=-1, tokenizer_type=tokeniser_type, stage="initial", vocab_size=vocab) | mp)
        # Create model: hyperparameters should be mostly identical to original music transformer paper
        model = MusicTransformer(
            tokenizer,
            n_layers=6,
            num_heads=8,
            d_model=512,
            dim_feedforward=1024,
            dropout=0.1,
            max_sequence=MAX_SEQUENCE_LEN,
            rpr=False
        ).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

        # Iterate over for the required number of epochs
        for n_epoch in range(1, N_EPOCHS_PER_VOCAB + 1):
            # Iterate through each subset
            for files_paths, subset_name in (
                    (midi_paths_train, "train"), (midi_paths_test, "test")
            ):
                # Split the MIDIs into chunks of sizes approximately 1024 tokens
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
                    batch_size=5,
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
                epoch_loss, epoch_acc, epoch_perp = [], [], []
                for batch in tqdm(dataloader, desc=f"{subset_name.title()}ing, vocab {vocab}, epoch {n_epoch}"):
                    # Backwards pass for training only
                    if subset_name == "train":
                        loss, accuracy, perplexity = forwards_pass(batch, model, tokenizer["PAD_None"])
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    # No backwards pass for testing
                    else:
                        with torch.no_grad():
                            loss, accuracy, perplexity = forwards_pass(batch, model, tokenizer["PAD_None"])
                    epoch_loss.append(loss.item())
                    epoch_acc.append(accuracy)
                    epoch_perp.append(perplexity.item())
                # Log to the console
                acc = np.mean(epoch_acc)
                norm_acc = acc / tokenizer.vocab_size
                perp = np.mean(epoch_perp)
                loss = np.mean(epoch_loss)
                print(f'Stage {subset_name}, vocab {vocab}, epoch {n_epoch}, '
                      f'loss: {loss:.3f}, accuracy: {acc:.3f}, normalised acc: {norm_acc:.3f}, perplexity: {perp:.3f}')
                res_dict = dict(
                    epoch=n_epoch,
                    stage=subset_name,
                    vocab_size=vocab,
                    loss=loss,
                    accuracy=acc,
                    normalised_accuracy=norm_acc,
                    perplexity=perp,
                    tokeniser_type=tokeniser_type
                )

                # Generating examples
                if subset_name == "test":
                    examples = generate_examples(dataloader, model)
                    # Randomly select one example to dump to MIDI
                    to_dump = random.choices(examples, k=1)[0]
                    out_path = os.path.join(
                        get_project_root(),
                        "outputs/sanity_check",
                        f"sanitycheck_{tokeniser_type}_{vocab}_{n_epoch}.mid"
                    )
                    tokenizer.decode(to_dump).dump_midi(out_path)
                    # Compute metrics for all the examples
                    gen_metrics = metrics.compute_metrics_for_sequences(examples, tokenizer)
                    res_dict = res_dict | gen_metrics
                results.append(res_dict)
                save_results(results, tokeniser_type, vocab)


if __name__ == "__main__":
    seed_everything()

    parser = ArgumentParser(description="Simple training script that allows different tokenisers to be evaluated")
    parser.add_argument(
        '-t', '--tokenizers',
        nargs='+',
        help='Tokenizer types to use',
        default=TOKENIZER_TYPES,
        type=str
    )
    parser.add_argument(
        '-v', '--vocab-sizes',
        nargs='+',
        help='Vocab sizes to use',
        default=VOCAB_SIZES,
        type=int
    )
    args = vars(parser.parse_args())
    tokenizers = args["tokenizers"]
    vs = args["vocab_sizes"]
    print(f'Using tokenizers {tokenizers}, vocab_sizes {vs}')
    for tk in tokenizers:
        main(tk, vs)
