#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate MIDI for CLaMP-DPO"""

import os

import torch
from loguru import logger
from symusic import TimeSignature, Tempo, Score
from torch.utils.data import DataLoader
from tqdm import tqdm

from jazz_style_conditioned_generation import utils, training
from jazz_style_conditioned_generation.data.tokenizer import CustomTSD

# Default config file for the generator
GENERATIVE_MODEL_CFG = (
    "finetuning-custom-tokenizer/"
    "finetuning_customtok_10msmin_lineartime_moreaugment_linearwarmup10k_5e5_thresh1e4patience4_batch4_1024seq_12l8h768d3072ff.yaml"
)


class ConditionTokenLoader:
    """Returns random pianist/genre tokens for use in generation"""

    def __init__(self, tokenizer, genre_ids: list[int] = None, pianist_ids: list[int] = None):
        self.tokenizer = tokenizer
        self.pianists = [i for i in sorted(tokenizer.vocab.keys()) if i.startswith("PIANIST")]
        self.genres = [i for i in sorted(tokenizer.vocab.keys()) if i.startswith("GENRES")]
        # Subset to get only required genres and pianists
        if genre_ids is not None:
            self.genres = [i for n, i in enumerate(self.genres) if n in genre_ids]
        else:
            self.genres = []
        if pianist_ids is not None:
            self.pianists = [i for n, i in enumerate(self.pianists) if n in pianist_ids]
        else:
            self.pianists = []
        # Combine everything together
        self.to_gen_from = sorted(self.pianists + self.genres)

    def __len__(self):
        return len(self.to_gen_from)

    def __getitem__(self, idx: int):
        # Make a random choice of either a genre or a pianist
        to_gen = self.to_gen_from[idx]
        # Assemble everything into a single list and add the BOS token after the selected condition token
        assembled = [self.tokenizer[to_gen], self.tokenizer["BOS_None"]]
        return dict(
            condition_ids=torch.tensor(assembled),
            generate_id=to_gen.split("_")[1].lower(),
        )


class ReinforceGenerateModule(training.TrainingModule):
    def __init__(self, genre_ids: list[int] = None, pianist_ids: list[int] = None, **training_kwargs):
        self.genre_ids = genre_ids
        self.pianist_ids = pianist_ids
        # Need to grab this and remove from kwargs before passing to the training module
        self.reinforce_cfg = training_kwargs.pop("reinforce_cfg", dict())
        self.policy_checkpoint_path = self.reinforce_cfg.pop("policy_model_checkpoint", None)
        # No MLFlow, we're not optimising anything
        training_kwargs["mlflow_cfg"]["use"] = False
        # This initialises our dataloaders, generative model, loads checkpoints, etc.
        super().__init__(**training_kwargs)
        # Whether we're tokenizing using ttype="Second" or ttype="Tick"
        self.ttype = "Second" if isinstance(self.tokenizer, CustomTSD) else "tick"

        # Set parameters for generation
        logger.info("----REINFORCEMENT LEARNING: GENERATING MIDIS----")
        self.generated_sequence_length = self.reinforce_cfg.get("generated_sequence_length", 1024)
        self.n_generations = self.reinforce_cfg.get("n_generations", 400)  # number of generations to make per track
        self.current_iteration = self.reinforce_cfg.get("current_iteration", 0)
        # By default, don't use temperature or top-p sampling
        self.temperature = self.reinforce_cfg.get("temperature", 1.0)
        self.top_p = self.reinforce_cfg.get("top_p", 1.0)

        logger.debug(f"For each of our {len(self.train_loader)} genre/pianist configurations, "
                     f"we'll generate {self.n_generations} tracks of {self.generated_sequence_length} tokens each. "
                     f"Generations will be stored in {self.generation_output_dir}.")

    def load_most_recent_checkpoint(self, weights_only: bool = True) -> None:
        # Load the checkpoint with the best validation loss (we don't care about optimizer/scheduler here)
        self.load_checkpoint(os.path.join(utils.get_project_root(), "checkpoints", self.policy_checkpoint_path),
                             weights_only=True)

    def create_dataloaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """We only want to create a single full-track dataloader"""
        train_loader = DataLoader(
            ConditionTokenLoader(tokenizer=self.tokenizer, genre_ids=self.genre_ids, pianist_ids=self.pianist_ids),
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

    def do_generation(self, condition_tokens: torch.Tensor) -> tuple[torch.Tensor, Score]:
        """Given condition tokens, generate MIDI and return tokens + decoded Score object"""
        # Need to be in evaluation mode for generation
        self.model.eval()
        # Through the model with desired top-p and temperature
        gen_i = self.model.generate(
            condition_tokens,
            target_seq_length=self.generated_sequence_length,
            top_p=self.top_p,
            temperature=self.temperature,
        )
        # Pad to the desired length if required
        if gen_i.size(1) < self.generated_sequence_length:
            gen_i = torch.nn.functional.pad(
                gen_i,
                (0, self.generated_sequence_length - gen_i.size(1)),
                value=self.tokenizer.pad_token_id
            )
        # Convert to a score, resample to desired sample rate, and set tempo/time signature
        gen_score = self.tokenizer.decode(gen_i.cpu()).resample(utils.TICKS_PER_QUARTER).to(self.ttype)
        gen_score.time_signatures = [TimeSignature(
            time=gen_score.time_signatures[0].time,
            numerator=utils.TIME_SIGNATURE,
            denominator=4,
            ttype=self.ttype
        )]
        gen_score.tempos = [Tempo(
            time=gen_score.tempos[0].time,
            qpm=utils.TEMPO,
            ttype=self.ttype
        )]
        return gen_i, gen_score

    def start(self):
        """Makes N generations for all required genre/pianist tokens"""
        self.model.eval()
        # Iterate over every track
        for batch in self.train_loader:
            condition_token = batch["generate_id"]
            # Iterate over all the generations we want to make
            for gen_idx in tqdm(range(self.n_generations), desc=f"Generating with token {condition_token}..."):
                fname = os.path.join(
                    self.generation_output_dir,
                    f"{batch['generate_id']}_iter{str(self.current_iteration).zfill(3)}_gen{str(gen_idx).zfill(3)}"
                )
                # Don't make the generation if it already exists as both a MIDI and pytorch tensor
                if os.path.isfile(fname + ".mid") and os.path.isfile(fname + ".pt"):
                    continue
                # Do the generation, get the token sequence + decoded score (we need to save both)
                condition_tokens = batch["condition_ids"]
                gen_tokens, gen_score = self.do_generation(condition_tokens)
                # Dump the MIDI and tensor to disk
                gen_score.dump_midi(fname + ".mid")
                torch.save(gen_tokens, fname + ".pt")


if __name__ == "__main__":
    import argparse

    utils.seed_everything(utils.SEED)
    # Parsing arguments from the command line interface
    parser = argparse.ArgumentParser(description="Generate MIDI files for reinforcement learning")
    parser.add_argument(
        "-c", "--config", default=GENERATIVE_MODEL_CFG, type=str,
        help="Path to config YAML file, relative to root folder of the project"
    )
    parser.add_argument(
        "-g", "--genres", nargs="+", type=int, default=None,
        help="IDs corresponding to genres to use in generation. Defaults to all genres."
    )
    parser.add_argument(
        "-p", "--pianists", nargs="+", type=int, default=None,
        help="IDs corresponding to pianists to use in generation. Defaults to all performers."
    )
    # Parse all arguments from the command line
    parser_args = vars(parser.parse_args())
    if not parser_args["config"]:
        raise ValueError("No config file specified")
    # Parse the config file
    cfg = training.parse_config_yaml(parser_args["config"])
    # Run training
    cls = ReinforceGenerateModule(genre_ids=parser_args["genres"], pianist_ids=parser_args["pianists"], **cfg)
    cls.start()  # no need for any mlflow here!
