#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for music transformer"""

import os
import unittest

import torch

from jazz_style_conditioned_generation import utils, metrics
from jazz_style_conditioned_generation.data.dataloader import create_padding_mask, DatasetMIDIConditionedFullTrack
from jazz_style_conditioned_generation.data.tokenizer import load_tokenizer, train_tokenizer
from jazz_style_conditioned_generation.encoders.music_transformer import MusicTransformer

# Create the model with default parameters
TOKENIZER = load_tokenizer()
MODEL = MusicTransformer(tokenizer=TOKENIZER).to(utils.DEVICE)
MODEL_RPR = MusicTransformer(tokenizer=TOKENIZER, rpr=True).to(utils.DEVICE)

TEST_MIDI_LONG = os.path.join(utils.get_project_root(), "tests/test_resources/test_midi1/piano_midi.mid")
TEST_MIDI_SHORT = os.path.join(utils.get_project_root(), "tests/test_resources/test_midi_repeatnotes.mid")


def handle_cuda_exceptions(f):
    """Skips a test when we get a CUDA out-of-memory error, allowing tests to run parallel with training runs."""

    def wrapper(*args, **kw):
        try:
            return f(*args, **kw)
        except torch.cuda.OutOfMemoryError:
            unittest.skip("Ignoring CUDA out of memory error!")

    return wrapper


class MusicTransformerTest(unittest.TestCase):

    @handle_cuda_exceptions
    def test_forward(self):
        def runner(model):
            model.train()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            # Get model parameters
            params = [np for np in model.named_parameters() if np[1].requires_grad]
            # Make a copy for later comparison
            initial_params = [(name, p.clone()) for (name, p) in params]
            # Create a dummy sequence of inputs: IDs, targets, and padding mask
            dummy_tensor = torch.randint(0, 100, (1, utils.MAX_SEQUENCE_LENGTH + 1)).to(utils.DEVICE)
            inp = dummy_tensor[:, :-1].to(utils.DEVICE)
            targ = dummy_tensor[:, 1:].to(utils.DEVICE)
            padding_mask = create_padding_mask(dummy_tensor, TOKENIZER.pad_token_id)[:, :-1].to(utils.DEVICE)
            # Forwards and backwards pass through the model
            logits = model(inp, targ, padding_mask)
            loss = metrics.cross_entropy_loss(logits, targ, TOKENIZER)
            opt.zero_grad()
            loss.backward()
            opt.step()
            # Expected shape of the logits should be (batch_size, sequence_length, vocab_size)
            expected = (1, utils.MAX_SEQUENCE_LENGTH, TOKENIZER.vocab_size)
            actual = tuple(logits.size())
            self.assertEqual(expected, actual)
            # Iterate through all the parameters in the model: they should have updated
            for (_, p0), (name, p1) in zip(initial_params, params):
                # If vars have changed, will return True; if not, will return False
                self.assertTrue(not torch.equal(p0.to(utils.DEVICE), p1.to(utils.DEVICE)))

        # We should do the same function for both RPR=True, RPR=False
        runner(MODEL)
        runner(MODEL_RPR)

    @handle_cuda_exceptions
    def test_generate(self):
        def runner(model):
            # Create a dummy tensor
            dummy_size = 128
            dummy_input = torch.randint(0, 100, (dummy_size,)).to(utils.DEVICE)
            # Generate with a target of double the dummy tensor size
            model.eval()
            generation = model.generate(dummy_input, target_seq_length=256)
            # The model may decide to terminate generation early, but we should still expect a longer sequence
            #  that what we passed in to it initially
            self.assertGreater(generation.size(1), dummy_size)

        # We should do the same function for both RPR=True, RPR=False
        runner(MODEL)
        # This test is flaky!
        runner(MODEL_RPR)

    @unittest.skipIf(os.getenv("REMOTE") == "true", "Skipping test on GitHub Actions")
    @handle_cuda_exceptions
    def test_evaluate(self):
        def runner(tokenizer):
            model = MusicTransformer(tokenizer=tokenizer).to(utils.DEVICE)
            for track in ds:
                # Unpack everything
                inputs = track["input_ids"].to(utils.DEVICE)
                targets = track["labels"].to(utils.DEVICE)
                mask = track["attention_mask"].to(utils.DEVICE)
                # Through the model
                tokens_loss = model.evaluate(inputs, targets, mask)
                # We'd expect the loss to be between 0 and an arbitrarily large value (the model hasn't been trained!)
                self.assertTrue(0. <= tokens_loss.item() <= 10.)

        # First, test without a trained tokenizer
        toker = load_tokenizer(tokenizer_str="midilike")
        # Create the dataset that returns full-length tracks
        ds = torch.utils.data.DataLoader(
            DatasetMIDIConditionedFullTrack(
                tokenizer=toker,
                files_paths=[TEST_MIDI_LONG, TEST_MIDI_SHORT],
                do_conditioning=False,  # no conditioning for now
                do_augmentation=False,
                max_seq_len=utils.MAX_SEQUENCE_LENGTH,
            ),
            batch_size=1,  # batch size MUST be set to one with this dataloader as output sequences have different len
            shuffle=False,
            drop_last=False,
        )
        runner(toker)
        # Second, test after training the tokenizer on the input track
        train_tokenizer(toker, [TEST_MIDI_LONG], vocab_size=500)
        runner(toker)

    def test_sulun_configuration(self):
        """Tests config in Sulun et al. (2022), Symbolic Music Generation Conditioned on Continuous-Valued Emotion"""
        tok = load_tokenizer()
        fps = [
            os.path.join(utils.get_project_root(), "tests/test_resources/test_midi1/piano_midi.mid"),
            os.path.join(utils.get_project_root(), "tests/test_resources/test_midi2/piano_midi.mid"),
            os.path.join(utils.get_project_root(), "tests/test_resources/test_midi3/piano_midi.mid"),
        ]
        tok.train(vocab_size=1007, files_paths=fps)  # this is just simply to get the reported vocabulary size
        model = MusicTransformer(
            tok,
            rpr=True,
            n_layers=20,
            num_heads=16,
            d_model=768,
            dim_feedforward=3072,
            dropout=0.1,
            max_seq_len=1216
        )
        n_params = utils.total_parameters(model)
        n_params_round = utils.base_round(n_params, 5000000)  # rounding to nearest 5 million
        self.assertTrue(n_params_round == 145000000)  # paper reports "about 145 million parameters"


if __name__ == '__main__':
    unittest.main()
