#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate continuations to the opening of Keith Jarrett's "The Koln Concert" in the style of different players"""

import os

import torch
from tqdm import tqdm

from jazz_style_conditioned_generation import generate, utils
from jazz_style_conditioned_generation.data.conditions import INCLUDE
from jazz_style_conditioned_generation.data.scores import load_score, preprocess_score
from jazz_style_conditioned_generation.training import parse_config_yaml

KOLN_OUTPUT_DIR = os.path.join(utils.get_project_root(), "outputs/generation/koln_generation")
N_GENERATIONS_PER_PIANIST = 10  # we'll generate this many examples per pianist


class KolnDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer):
        self.pianists = INCLUDE["pianist"]
        self.tokenizer = tokenizer
        # Load in the Koln midi as a score, preprocess, and convert to a token sequence
        koln_path = os.path.join(utils.get_project_root(), "references/koln-intro.MID")
        koln_score = preprocess_score(load_score(koln_path, as_seconds=True))
        self.koln_toks = self.tokenizer(koln_score)[0].ids

    def __len__(self) -> int:
        return len(self.pianists)

    def __getitem__(self, idx: int):
        # Get the current pianist and convert to a token
        pianist = self.pianists[idx]
        pianist_tok = self.tokenizer.vocab["PIANIST_" + pianist.replace(" ", "")]
        # Combine the pianist token, START, and the Koln tokens
        combined_toks = torch.tensor([pianist_tok, self.tokenizer.vocab["BOS_None"]] + self.koln_toks)
        return pianist, combined_toks


class KolnKontinuer(generate.GenerateModule):
    def get_test_loader(self):
        # Overwrite our test loading function
        return torch.utils.data.DataLoader(
            KolnDataset(tokenizer=self.tokenizer),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            drop_last=False
        )

    def start(self):
        for pianist, tokseq in self.test_loader:
            pianist = pianist[0].lower().replace(" ", "")
            tokseq = tokseq.squeeze(0)
            for n in tqdm(range(N_GENERATIONS_PER_PIANIST), desc=f"Generating for {pianist}"):
                n = str(n).zfill(3)
                self.model.eval()
                with torch.no_grad():
                    gen_out = self.model.generate(
                        tokseq,
                        target_seq_length=self.sequence_len,
                        top_p=self.top_p,
                        temperature=self.temperature
                    )
                # Detokenize the output
                gen_out = self.tokenizer(gen_out.cpu())
                # Dump the MIDI (always)
                out_fp = os.path.join(KOLN_OUTPUT_DIR, f"koln_{pianist}_{n}.mid")
                gen_out.dump_midi(out_fp)


if __name__ == "__main__":
    import argparse

    utils.seed_everything(utils.SEED)

    # Parsing arguments from the command line interface
    parser = argparse.ArgumentParser(description="Generate continuations to the opening of the Koln Concert")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path to config YAML file for trained model",
        default=("finetuning-customtok-plateau/"
                 "finetuning_customtok_10msmin_lineartime_moreaugment"
                 "_init6e5reduce10patience5_batch4_1024seq_12l8h768d3072ff.yaml")
    )
    # Parse all arguments from the provided YAML file
    args = vars(parser.parse_args())
    if not args:
        raise ValueError("No config file specified")
    generate_kws = parse_config_yaml(args['config'])
    # No MLFlow required for generation
    generate_kws["mlflow_cfg"]["use"] = False
    generate_kws["_generate_only"] = True  # avoids running our preprocessing for training + validation data
    # Prevents us from having to create the dataset from scratch
    generate_kws["test_dataset_cfg"]["n_clips"] = 1

    km = KolnKontinuer(**generate_kws, sequence_len=1024, temperature=1.0, top_p=1.0)
    km.start()
