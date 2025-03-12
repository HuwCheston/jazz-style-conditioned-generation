#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate using a model trained from running training.py"""

import os

import torch
from loguru import logger

from jazz_style_conditioned_generation import utils
from jazz_style_conditioned_generation.data.scores import load_score, preprocess_score
from jazz_style_conditioned_generation.training import TrainingModule, parse_config_yaml

OUTPUT_DIR = os.path.join(utils.get_project_root(), "outputs/generation")


def main(
        primer: str = None,
        primer_tokens: int = 128,
        sequence_len: int = 1024,
        top_p: float = 0.92,
        top_k: int = 5,
        output_dir: str = OUTPUT_DIR,
        save_wav: bool = True,
        **generate_kwargs
) -> None:
    # Sanity checks
    assert primer_tokens < sequence_len, "Primer tokens must be smaller than desired sequence length"
    assert 0. < top_p <= 1., "Top-p must be between 0 and 1"
    assert 0 <= top_k, "Top-k must be positive"

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
        sc = load_score(primer_fp)
        # Do our preprocessing
        sc = preprocess_score(sc)
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
        # No need to convert to a tensor or preprocess as this is already done in the dataloader
        tokseq = batch["input_ids"][0, :primer_tokens]
    # Finally, we can pass through the model to get the output
    logger.info("Starting generation...")
    with torch.no_grad():
        gen_out = tm.model.generate(tokseq, target_seq_length=sequence_len, top_p=top_p, top_k=top_k)
    logger.info(f"Generation finished with length {gen_out.size(1)}, saving output...")
    # Convert the generated token indices back to a Score
    tok_out = tm.tokenizer(gen_out.to("cpu"))
    # (post)-process the generated score
    tok_out = preprocess_score(tok_out)
    # Get the filepaths to save the generation to
    out_fp = os.path.join(output_dir, f"generation_{utils.now()}")
    # Save the midi + wav file (if required)
    tok_out.dump_midi(f"{out_fp}.mid")
    if save_wav:
        utils.synthesize_score(tok_out, f"{out_fp}.wav")
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
        help="Path to config YAML file for trained model"
    )
    parser.add_argument(
        "-m",
        "--primer-midi",
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
    parser.add_argument(
        "-l",
        "--sequence-len",
        default=utils.MAX_SEQUENCE_LENGTH,
        type=int,
        help="Total length of the sequence to generate (including primer), must be larger than number of primer tokens"
    )
    parser.add_argument(
        "-p",
        "--top-p",
        default=0.92,
        type=float,
        help="Top-p value to use in nucleus sampling (defaults to 0.92). Must be between 0 and 1."
    )
    parser.add_argument(
        "-k",
        "--top-k",
        default=5,
        type=int,
        help="Top-k value to use in nucleus sampling (defaults to 5). Must be positive."
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=OUTPUT_DIR,
        type=str,
        help="Directory to save the output to, defaults to ./outputs/generation/."
    )
    parser.add_argument(
        "-w",
        "--save-wav",
        default=True,
        type=utils.string_to_bool,
        help="If true, will save a synthesised WAV file to the output directory."
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
    main(
        primer=args["primer_midi"],
        primer_tokens=args["n_primer_tokens"],
        sequence_len=args["sequence_len"],
        top_p=args["top_p"],
        top_k=args["top_k"],
        output_dir=args["output_dir"],
        save_wav=args["save_wav"],
        **generate_kws
    )
