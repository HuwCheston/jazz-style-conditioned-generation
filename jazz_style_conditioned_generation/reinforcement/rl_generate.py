#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate MIDI for CLaMP-DPO"""

import os

import torch
from loguru import logger
from symusic import TimeSignature, Tempo
from torch.utils.data import DataLoader
from tqdm import tqdm

from jazz_style_conditioned_generation import utils, training

# Default config file for the generator
GENERATIVE_MODEL_CFG = (
    "reinforcement-clamp-ppo/"
    "music_transformer_rpr_tsd_nobpe_conditionsmall_augment_schedule_10l8h_clampppo_2e6_TEST.yaml"
)


class ClampGenerationLoader:
    """Returns random pianist/genre tokens for use in generation"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pianists = [i for i in tokenizer.vocab.keys() if i.startswith("PIANIST")]
        self.genres = [i for i in tokenizer.vocab.keys() if i.startswith("GENRES")]
        # We want to generate from every pianist and every genre
        self.to_gen_from = self.pianists + self.genres

    def __len__(self):
        return len(self.pianists) + len(self.genres)

    def __getitem__(self, idx: int):
        # Make a random choice of either a genre or a pianist
        to_gen = self.to_gen_from[idx]
        # Assemble everything into a single list and add the BOS token after the selected condition token
        assembled = [self.tokenizer[to_gen], self.tokenizer["BOS_None"]]
        return dict(
            condition_ids=torch.tensor(assembled),
            generate_id=to_gen.split("_")[1].lower(),
        )


class ClampGenerateModule(training.FineTuningModule):
    def __init__(self, **training_kwargs):
        # Need to grab this and remove from kwargs before passing to the training module
        self.reinforce_cfg = training_kwargs.pop("reinforce_cfg", dict())
        # No MLFlow, we're not optimising anything
        training_kwargs["mlflow_cfg"]["use"] = False
        # This initialises our dataloaders, generative model, loads checkpoints, etc.
        super().__init__(**training_kwargs)

        # Set parameters for generation
        logger.info("----REINFORCEMENT LEARNING: GENERATING MIDIS----")
        self.generated_sequence_length = self.reinforce_cfg.get("generated_sequence_length", utils.MAX_SEQUENCE_LENGTH)
        self.n_generations = self.reinforce_cfg.get("n_generations", 1000)  # number of generations to make per track
        self.current_iteration = self.reinforce_cfg.get("current_iteration", 0)

        logger.debug(f"For each of our {len(self.train_loader)} training tracks, "
                     f"we'll generate {self.n_generations} tracks of {self.generated_sequence_length} tokens.")

    def create_dataloaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """We only want to create a single full-track dataloader"""
        # Create training dataset loader: uses FULL tracks!
        train_loader = DataLoader(
            ClampGenerationLoader(tokenizer=self.tokenizer),
            batch_size=1,  # have to use a batch size of one for this class
            shuffle=False,  # don't want to shuffle either for this one
            drop_last=False,
            collate_fn=lambda x: {k: v if isinstance(v, str) else v.to(utils.DEVICE) for k, v in x[0].items()}
        )
        return train_loader, None, None

    @property
    def generation_output_dir(self):
        fpath = os.path.join(
            utils.get_project_root(),
            "data/rl_generations",
            f"{self.experiment}/{self.run}"
        )
        if not os.path.isdir(fpath):
            os.makedirs(fpath)
        return fpath

    def start(self):
        """Makes generations"""
        self.model.eval()
        # Iterate over every track
        for batch in self.train_loader:
            condition_token = batch["generate_id"]
            # Iterate over all the generations we want to make
            for gen_idx in tqdm(range(self.n_generations), desc=f"Generating with token {condition_token}..."):
                fname = os.path.join(
                    self.generation_output_dir,
                    f"{batch['generate_id']}_iter{str(self.current_iteration).zfill(3)}_gen{str(gen_idx).zfill(3)}.mid"
                )
                # Don't make the generation if it already exists
                if os.path.isfile(fname):
                    continue
                # Do the generation
                condition_tokens = batch["condition_ids"]
                gen_i = self.model.generate(condition_tokens, target_seq_length=self.generated_sequence_length)
                # Pad to the desired length if required
                if gen_i.size(1) < self.generated_sequence_length:
                    gen_i = torch.nn.functional.pad(
                        gen_i,
                        (0, self.generated_sequence_length - gen_i.size(1)),
                        value=self.tokenizer.pad_token_id
                    )
                # Convert to a score, resample to desired sample rate, and set tempo/time signature
                gen_score = self.tokenizer.decode(gen_i.cpu()).resample(utils.TICKS_PER_QUARTER)
                gen_score.time_signatures = [TimeSignature(
                    time=gen_score.time_signatures[0].time,
                    numerator=utils.TIME_SIGNATURE,
                    denominator=4,
                    ttype="tick"
                )]
                gen_score.tempos = [Tempo(
                    time=gen_score.tempos[0].time,
                    qpm=utils.TEMPO,
                    ttype="tick"
                )]
                # Dump to the disk
                gen_score.dump_midi(fname)


if __name__ == "__main__":
    import argparse

    utils.seed_everything(utils.SEED)
    # Parsing arguments from the command line interface
    parser = argparse.ArgumentParser(description="Generate MIDI files for reinforcement learning")
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
    training.main(training_kws=cfg, trainer_cls=ClampGenerateModule, config_fpath=parser_args["config"])
