#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate a trained model with the performer identification ResNet"""

import os

import numpy as np
import torch
from loguru import logger
from symusic import Score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from jazz_style_conditioned_generation import utils
from jazz_style_conditioned_generation.data.scores import load_score
from jazz_style_conditioned_generation.encoders.resnet import load_performer_identifier, CLASS_MAPPING

RESNET = load_performer_identifier()
# These parameters are copied from deep-pianist-identification repo
CLIP_LENGTH = 30  # ResNet was trained on 30 second chunks
FPS = 100  # Transcription model uses 100 frames-per-second


def score_to_piano_roll(score: Score) -> torch.Tensor:
    """Converts a symusic.Score object to a piano roll with shape (batch, channel, height, width)"""
    # Convert to a piano roll
    roll = score.tracks[0].pianoroll(
        modes=["frame"],  # we don't care about separate onset/offset rolls
        pitch_range=(utils.MIDI_OFFSET, utils.MIDI_OFFSET + utils.PIANO_KEYS),  # gives us desired height
        encode_velocity=True
    )
    downsampled = roll[:, :, ::10]  # downsamples from 1 column == 1 ms -> 1 column == 10 ms
    # Pads to shape (channel, 88, 3000), as used in performer identification model initially
    desired_width = CLIP_LENGTH * FPS
    if downsampled.shape[-1] < desired_width:
        clip = np.pad(
            downsampled,
            (
                (0, 0),
                (0, 0),
                (0, desired_width - downsampled.shape[-1])
            ),
            mode="constant",
            constant_values=0.
        )
    # Truncate from end to get shape (channel, 88, 3000)
    else:
        clip = downsampled[:, :, -desired_width:]
    # Normalize to within the range (0, 1)
    normalized = (clip - np.min(clip)) / (np.max(clip) - np.min(clip))
    return torch.tensor(normalized, dtype=torch.float32)


class GeneratedMIDILoader(Dataset):
    """Dataloader for generated MIDI files: returns piano rolls + target classes"""

    def __init__(self, files_paths: list[str], ):
        self.files_paths = list(self.get_valid_filepaths(files_paths))

    @staticmethod
    def get_valid_filepaths(files_paths):
        """Get tracks generated using prompts associated with valid pianists"""
        for track in files_paths:
            # Get the name of the pianist from the file path
            pianist = track.split(os.path.sep)[-1].split("_")[0]
            # Get the class IDX of the pianist, according to the mapping used to train the ResNet
            pianist_cls = [
                v for k, v in CLASS_MAPPING.items()
                if utils.remove_punctuation(k).lower().replace(" ", "") == pianist
            ]
            # Skip over if we have no matches (or more than one match: should never happen?)
            if len(pianist_cls) == 0 or len(pianist_cls) > 1:
                continue
            yield track, pianist_cls[0]

    def __len__(self) -> int:
        return len(self.files_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        track, pianist_cls = self.files_paths[index]
        # Load the generated MIDI up as a symusic.Score object
        score = load_score(track)
        # Convert the score to a piano roll
        proll = score_to_piano_roll(score)
        return proll, pianist_cls


class PerformerIDEvaluator:
    def __init__(self, midi_path: str, batch_size: int = 8):
        # This path must contain files in
        self.midi_path = midi_path
        self.midis = [os.path.join(self.midi_path, midi) for midi in os.listdir(self.midi_path)]
        utils.validate_paths(self.midis, expected_extension=".mid")

        # Initialise the parent module: this will load checkpoints, create the transformer model, etc.
        logger.info("----RUNNING EVALUATION WITH PRE-TRAINED PERFORMER ID RESNET----")
        self.generated_dataloader = DataLoader(
            GeneratedMIDILoader(self.midis),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False
        )

    def start(self):
        all_accs = []
        # Iterate over generated piano rolls and actual class
        for prolls, pianist_cls in tqdm(self.generated_dataloader, desc="Processing generated MIDI files..."):
            # Set all devices correctly
            prolls = prolls.to(utils.DEVICE)
            pianist_cls = pianist_cls.to(utils.DEVICE)
            # Forward pass through the model
            with torch.no_grad():
                resnet_logits = RESNET(prolls)
            # Compute predicted class by softmaxing -> argmaxing
            resnet_smaxed = torch.nn.functional.softmax(resnet_logits, dim=-1)
            predicted_pianist_cls = torch.argmax(resnet_smaxed, dim=-1)
            # Compute accuracy for batch as proportion of correct predictions
            batch_accuracy = (pianist_cls == predicted_pianist_cls).sum() / pianist_cls.size(-1)
            all_accs.append(batch_accuracy.item())
        # Compute summary statistics and log
        mean_acc = np.mean(all_accs)
        std_acc = np.std(all_accs)
        logger.info(f"Finished! Mean accuracy: {mean_acc:.3f}, std {std_acc:.3f}")


if __name__ == "__main__":
    import argparse

    utils.seed_everything(utils.SEED)
    # Parsing arguments from the command line interface
    parser = argparse.ArgumentParser(description="Evaluate a model with the performer identification ResNet")
    parser.add_argument(
        "-m", "--midi-path", type=str,
        help="Path to directory containing generated MIDI files for use in evaluation"
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=8,
        help="Batch size to use when processing generated MIDI outputs"
    )
    # Parse all arguments from the command line
    parser_args = vars(parser.parse_args())
    # Create the performer ID evaluator and run evaluation
    pide = PerformerIDEvaluator(midi_path=parser_args["midi_path"], batch_size=parser_args["batch_size"])
    pide.start()
