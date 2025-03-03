#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate using a model trained from running training.py"""

import os

import torch
from loguru import logger
from symusic import Score

from jazz_style_conditioned_generation import utils
from jazz_style_conditioned_generation.training import TrainingModule, parse_config_yaml


def main(primer: str = None, primer_tokens: int = None, **generate_kwargs):
    tm = TrainingModule(**generate_kwargs)
    # Model must be in evaluation mode
    tm.model.eval()
    # Load the best checkpoint
    tm.load_checkpoint(os.path.join(tm.checkpoint_dir, 'validation_best.pth'))
    # If we've specified a primer file
    if primer is not None:
        primer_fp = os.path.join(utils.get_project_root(), primer)
        logger.info(f"Generating using primer MIDI at {primer_fp}")
        # Load the primer file
        assert os.path.isfile(primer_fp), f"Could not find MIDI file at {primer_fp}!"
        # Convert to a symusic object
        sc = Score(primer_fp)
        # Pass the score through the tokenizer
        tokseq = tm.tokenizer(sc)[0].ids
        # Subset the tokens to get the required number and convert to a tensor
        tokseq = torch.tensor(tokseq[:primer_tokens])
    # Otherwise, we can draw a random file from the test dataset
    else:
        logger.info("Generating using a random item from test split")
        # Get the first batch
        batch = next(iter(tm.test_loader))
        # Subset to get the required number of tokens from the first item in the batch
        # No need to convert to a tensor as it already is one
        tokseq = batch["input_ids"][0, :primer_tokens]
    # Finally, we can pass through the model to get the output
    logger.info("Starting generation...")
    with torch.no_grad():
        gen_out = tm.model.generate(tokseq, target_seq_length=512)
    tok_out = tm.tokenizer(gen_out.to("cpu"))
    # Dump the output
    tok_out.dump_midi("output.mid")
    logger.info("Done!")


if __name__ == "__main__":
    import argparse

    # Seed everything for reproducible results
    utils.seed_everything(utils.SEED)

    # Parsing arguments from the command line interface
    parser = argparse.ArgumentParser(description="Generate using a trained model")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "-p",
        "--primer",
        default=None,
        type=str,
        help="Path to primer MIDI file to use in generation"
    )
    parser.add_argument(
        "-n",
        "--n-primer-tokens",
        default=128,
        type=int,
        help="Number of primer tokens to use in generation"
    )
    # Parse all arguments from the provided YAML file
    args = vars(parser.parse_args())
    if not args:
        raise ValueError("No config file specified")
    generate_kws = parse_config_yaml(args['config'])
    # No MLFlow required for generation
    generate_kws["mlflow_cfg"]["use"] = False
    # Prevents us from having to create the dataset from scratch
    generate_kws["test_dataset_cfg"]["n_clips"] = 1
    main(primer=args["primer"], primer_tokens=args["n_primer_tokens"], **generate_kws)
