#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for music transformer"""

import unittest

import torch
from miditok import REMI

from jazz_style_conditioned_generation import utils
from jazz_style_conditioned_generation.data.dataloader import create_padding_mask
from jazz_style_conditioned_generation.encoders.music_transformer import MusicTransformer

# Create the model with default parameters
TOKENIZER = REMI()
MODEL = MusicTransformer(tokenizer=TOKENIZER).to(utils.DEVICE)
MODEL_RPR = MusicTransformer(tokenizer=TOKENIZER, rpr=True).to(utils.DEVICE)


class MusicTransformerTest(unittest.TestCase):
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
            out = model(inp, targ, padding_mask)
            loss = out.loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            # Expected shape of the logits should be (batch_size, sequence_length, vocab_size)
            expected = (1, utils.MAX_SEQUENCE_LENGTH, TOKENIZER.vocab_size)
            actual = tuple(out.logits.size())
            self.assertEqual(expected, actual)
            # Iterate through all the parameters in the model: they should have updated
            for (_, p0), (name, p1) in zip(initial_params, params):
                # If vars have changed, will return True; if not, will return False
                self.assertTrue(not torch.equal(p0.to(utils.DEVICE), p1.to(utils.DEVICE)))

        # We should do the same function for both RPR=True, RPR=False
        runner(MODEL)
        runner(MODEL_RPR)

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
        runner(MODEL_RPR)


if __name__ == '__main__':
    unittest.main()
