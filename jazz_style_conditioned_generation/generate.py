#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate using a model trained from running training.py"""

import os
import random

import torch
from loguru import logger
from torch.utils.data import DataLoader

from jazz_style_conditioned_generation import utils
from jazz_style_conditioned_generation.data.conditions import (
    MAX_PIANIST_TOKENS_PER_TRACK,
    MAX_GENRE_TOKENS_PER_TRACK,
    get_genre_tokens,
    get_pianist_tokens,
    get_time_signature_token,
    get_tempo_token,
    get_recording_year_token
)
from jazz_style_conditioned_generation.data.dataloader import DatasetMIDIConditionedFullTrack
from jazz_style_conditioned_generation.encoders.music_transformer import DEFAULT_TEMPERATURE, DEFAULT_TOP_P
from jazz_style_conditioned_generation.training import TrainingModule, parse_config_yaml

OUTPUT_DIR = os.path.join(utils.get_project_root(), "outputs/generation")


class GenerateDataset(DatasetMIDIConditionedFullTrack):
    """Custom dataset designed for use in generation only"""

    def __init__(
            self,
            tokenizer,
            files_paths: list[str],
            max_seq_len: int,
            do_augmentation: bool = False,
            do_conditioning: bool = True,
            n_clips: int = None,
            max_pianist_tokens: int = MAX_PIANIST_TOKENS_PER_TRACK,
            max_genre_tokens: int = MAX_GENRE_TOKENS_PER_TRACK,
            custom_pianists: list[str] = None,
            custom_genres: list[str] = None,
            custom_time_signature: int = None,
            custom_recording_year: int = None,
            custom_tempo: int = None,
            use_track_tokens: bool = True,
    ):
        # We have to set these attributes before we call super().__init__()
        #  This is because we need these attributes inside our overridden `get_conditioning_tokens`
        #  which itself is called inside super().preload_data()
        self.custom_pianists = custom_pianists
        self.custom_genres = custom_genres
        self.custom_time_signature = custom_time_signature
        self.custom_recording_year = custom_recording_year
        self.custom_tempo = custom_tempo
        self.use_track_tokens = use_track_tokens
        super().__init__(
            tokenizer,
            files_paths,
            max_seq_len,
            do_augmentation,
            do_conditioning,
            n_clips,
            max_pianist_tokens,
            max_genre_tokens
        )

    def get_conditioning_tokens(self, metadata: dict) -> list[str]:
        """Overrides base methods, allows use of condition tokens not contained in a track's metadata"""

        def getter(custom_attr, name, func) -> list[str]:
            # If we have the custom attribute, use that
            if custom_attr:
                return [func(custom_attr, self.tokenizer)]
            # Otherwise, fall back to the value assigned to the track
            elif name in metadata.keys() and self.use_track_tokens:
                return [func(metadata[name], self.tokenizer)]
            # Otherwise, don't use this token
            else:
                return []

        # Grab the custom genre tokens
        if self.custom_genres:
            finalised_genre_tokens = [
                f"GENRES_{utils.remove_punctuation(g).replace(' ', '')}"
                for g in self.custom_genres
            ]
        # Otherwise, if we haven't passed these in, use those assigned to the track
        elif self.use_track_tokens:
            finalised_genre_tokens = get_genre_tokens(metadata, self.tokenizer)
        else:
            finalised_genre_tokens = []

        # Grab the pianist tokens
        if self.custom_pianists:
            finalised_pianist_tokens = [
                f"PIANIST_{utils.remove_punctuation(g).replace(' ', '')}"
                for g in self.custom_pianists
            ]
        elif self.use_track_tokens:
            finalised_pianist_tokens = get_pianist_tokens(metadata, self.tokenizer)
        else:
            finalised_pianist_tokens = []

        # Grab time signature, tempo, and year tokens
        extra_tokens = [
            *finalised_genre_tokens,
            *finalised_pianist_tokens,
            *getter(self.custom_recording_year, "recording_year", get_recording_year_token),
            *getter(self.custom_tempo, "tempo", get_tempo_token),
            *getter(self.custom_time_signature, "time_signature", get_time_signature_token),
        ]
        logger.info(f"... generating using condition tokens: {', '.join(i for i in extra_tokens)}")
        return [self.tokenizer[et] for et in extra_tokens]


class GenerateModule(TrainingModule):
    def __init__(self, **kwargs):
        # Get the arguments from the kwargs
        self.primer_file: str = kwargs.pop("primer", None)
        self.sequence_len = kwargs.pop("sequence_len", utils.MAX_SEQUENCE_LENGTH)
        self.primer_tokens = kwargs.pop("primer_tokens", self.sequence_len // 4)
        self.top_p = kwargs.pop("top_p", DEFAULT_TOP_P)
        self.temperature = kwargs.pop("temperature", DEFAULT_TEMPERATURE)
        self.output_dir = kwargs.pop("output_dir", OUTPUT_DIR)
        self.save_wav = kwargs.pop("save_wav", True)

        # Custom generate arguments
        self.custom_pianists = kwargs.pop("pianists", None)
        self.custom_genres = kwargs.pop("genres", None)
        self.custom_time_signature = kwargs.pop("time_signature", None)
        self.custom_recording_year = kwargs.pop("recording_year", None)
        self.custom_tempo = kwargs.pop("tempo", None)
        self.use_track_tokens = kwargs.pop("use_track_tokens", False)

        # Sanity checks
        assert self.primer_tokens < self.sequence_len, "Primer tokens must be smaller than desired sequence length"
        assert 0. < self.top_p <= 1., "Top-p must be between 0 and 1"
        kwargs.pop("pretrained_checkpoint_path", None)  # remove from pretrained models

        # Initialise the training module: this will SKIP creating dataloaders
        super().__init__(**kwargs)

        # Make a random file selection from the test split if we haven't passed in a primer file
        if self.primer_file is None:
            self.primer_file = random.choices(self.track_splits["test"], k=1)[0]
        utils.validate_paths([self.primer_file], expected_extension=".mid")

        # Create a dataloader solely with only our primer file
        self.test_loader = DataLoader(
            GenerateDataset(
                tokenizer=self.tokenizer,
                files_paths=[self.primer_file],
                max_seq_len=utils.MAX_SEQUENCE_LENGTH,
                do_augmentation=False,
                do_conditioning=True,
                custom_pianists=self.custom_pianists,
                custom_genres=self.custom_genres,
                custom_time_signature=self.custom_time_signature,
                custom_recording_year=self.custom_recording_year,
                custom_tempo=self.custom_tempo,
                use_track_tokens=self.use_track_tokens
            ),
            batch_size=1,  # have to use a batch size of one for this class
            shuffle=False,  # don't want to shuffle either for this one
            drop_last=False,
        )

    @property
    def num_training_steps(self) -> int:
        """Little hack when we don't need a scheduler"""
        return 1

    def create_dataloaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Little hack that avoids needing to do costly preloading of all dataloaders inside __init__"""
        return None, None, None

    def do_generation(self) -> torch.Tensor:
        # Get the first batch from our dataloader
        batch = next(iter(self.test_loader))
        # Subset to get the required number of tokens from the first item in the batch
        # No need to convert to a tensor or preprocess as this is already done inside the dataloader
        tokseq = batch["input_ids"][0, :self.primer_tokens]
        # Do the generation
        self.model.eval()
        with torch.no_grad():
            gen_out = self.model.generate(
                tokseq,
                target_seq_length=self.sequence_len,
                top_p=self.top_p,
                temperature=self.temperature
            )
        # Convert the generated token indices back to a Score
        return gen_out.cpu()

    def start(self):
        # Do the generation
        logger.info(f"Starting generation using primer MIDI {self.primer_file}...")
        tok_out = self.do_generation()
        logger.info(f"Generation finished with tokens {tok_out.size(1)}, saving output...")
        # Detokenize the output
        gen_out = self.tokenizer(tok_out)
        out_fp = os.path.join(self.output_dir, f"generation_{utils.now()}")
        # Dump the MIDI (always)
        gen_out.dump_midi(out_fp + ".mid")
        # Dump the WAV (optional)
        if self.save_wav:
            utils.synthesize_score(gen_out, out_fp + ".wav")
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
        default=DEFAULT_TOP_P,
        type=float,
        help=f"Top-p value to use in nucleus sampling (defaults to {DEFAULT_TOP_P}). Must be between 0 and 1."
    )
    parser.add_argument(
        "-t",
        "--temperature",
        default=DEFAULT_TEMPERATURE,
        type=float,
        help=f"Temperature value to use in scaling the probability distribution (defaults to {DEFAULT_TEMPERATURE})."
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

    # These arguments condition the generation
    parser.add_argument(
        "--pianist",
        action="append",
        type=str,
        help="Names of custom pianists to use in conditioning the generated MIDI. "
             "Multiple values are accepted, e.g. `--pianist 'Brad Mehldau' --pianist 'Keith Jarrett'",
        required=False
    )
    parser.add_argument(
        "--genre",
        action="append",
        type=str,
        help="Names of custom genres to use in conditioning the generated MIDI. "
             "Multiple values are accepted, e.g. `--genre 'African' --genre 'Blues' --genre 'Fusion'",
        required=False
    )
    parser.add_argument(
        "--time-signature",
        type=int,
        help="Custom time signature to use in conditioning the generated MIDI, in number of quarter-note beats. "
             "Only one value is accepted, e.g. `--time-signature 4`. Must be either 3 or 4"
    )
    parser.add_argument(
        "--tempo",
        type=int,
        help="Custom tempo to use in conditioning the generated MIDI, in number of quarter-note beats-per-minute. "
             "Only one value is accepted, e.g. `--tempo 260`"
    )
    parser.add_argument(
        "--recording-year",
        type=int,
        help="Custom recording year to use in conditioning the generated MIDI. "
             "Only one value is accepted, e.g. `--recording-year 1960`"
    )
    parser.add_argument(
        "--use-track-tokens",
        type=utils.string_to_bool,
        default=False,
        help="Use the tokens assigned to primer when no custom tokens are provided for e.g., genre, pianist, tempo. "
             "Defaults to False."
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

    # Create the module and do the generation
    gm = GenerateModule(
        primer=args["primer_midi"],
        primer_tokens=args["n_primer_tokens"],
        sequence_len=args["sequence_len"],
        top_p=args["top_p"],
        temperature=args["temperature"],
        output_dir=args["output_dir"],
        save_wav=args["save_wav"],
        pianists=args["pianist"],
        genres=args["genre"],
        time_signature=args["time_signature"],
        tempo=args["tempo"],
        recording_year=args["recording_year"],
        use_track_tokens=args["use_track_tokens"],
        **generate_kws
    )
    gm.start()
