#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for reinforcement learning modules"""

import os
import unittest

import torch
from symusic import Score

from jazz_style_conditioned_generation import utils, training
from jazz_style_conditioned_generation.reinforcement.rl_generate import ReinforceGenerateModule
from jazz_style_conditioned_generation.reinforcement.rl_train import ReinforceTrainModule, GroundTruthDataset

# Config file for the generator
GENERATIVE_MODEL_CFG = os.path.join(
    utils.get_project_root(),
    "config",
    "reinforcement-clamp-ppo",
    "music_transformer_rpr_tsd_nobpe_conditionsmall_augment_schedule_10l8h_clampppo_2e6_TEST.yaml"
)


# Need to skip on GitHub actions because of CLaMP requirement
@unittest.skipIf(os.getenv("REMOTE") == "true", "Skipping test on GitHub Actions")
class ReinforceTrainTest(unittest.TestCase):
    cfg_parsed = training.parse_config_yaml(GENERATIVE_MODEL_CFG)
    RT = ReinforceTrainModule(**cfg_parsed)

    def test_sort_generations(self):
        # Create some simple "generations"
        # We set this up in such a way that the worst generations will have the smallest numbers in the tensor
        # Vice versa, the best generations will have the largest numbers
        gens = [(torch.tensor([[i, i + 1, i + 2, i + 3]]), i / 10) for i in range(10)]
        best, worst = self.RT.sort_generations(gens)
        # We're only keeping 10% of the generations for best/worst, so each tensor should be of length 1
        self.assertTrue(best.size(0) == worst.size(0) == 1)
        # Best tensor should be the last one in the list
        self.assertTrue(torch.equal(best, gens[-1][0]))
        # Worst tensor should be the first one in the list
        self.assertTrue(torch.equal(worst, gens[0][0]))

    def test_compute_log_probs_grad(self):
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]])
        self.RT.model.eval()
        # Test with no_grad=True
        lprobs_ng = self.RT.compute_log_probs(self.RT.model, input_ids, no_grad=True)
        self.assertFalse(lprobs_ng.requires_grad)
        self.assertTrue(lprobs_ng.item() < 0.)  # log probs are negative
        # Test with no_grad=False
        lprobs_rg = self.RT.compute_log_probs(self.RT.model, input_ids, no_grad=False)
        self.assertTrue(lprobs_rg.requires_grad)
        self.assertTrue(lprobs_rg.item() < 0.)  # log probs are negative
        # Should both be equivalent
        self.assertEqual(lprobs_ng.item(), lprobs_rg.item())

    def test_compute_log_probs_mask(self):
        input_ids_nomask = torch.tensor([[1, 2, 3, 4, 5]])
        input_ids_withmask = torch.tensor([[1, 2, 3, 4, 0]])
        self.RT.model.eval()
        # With a mask token, we'd expect the sum of all log probs to be smaller than without a mask token
        lprobs_nomask = self.RT.compute_log_probs(self.RT.model, input_ids_nomask, no_grad=True)
        lprobs_withmask = self.RT.compute_log_probs(self.RT.model, input_ids_withmask, no_grad=True)
        self.assertLess(lprobs_withmask.item(), lprobs_nomask.item())

    def test_dpo_loss(self):
        # Test with a batch size of 2
        best_b2, worst_b2 = torch.randint(1, 10, (2, 10)), torch.randint(1, 10, (2, 10))
        # Compute loss
        loss = self.RT.compute_dpo_loss(best_b2, worst_b2)
        # Loss should be within the range [0, inf]
        self.assertTrue(0 <= loss.item() < torch.inf)
        # Should require grad
        self.assertTrue(loss.requires_grad)

    def test_ground_truth_dataset(self):
        resources_root = os.path.join(utils.get_project_root(), "tests/test_resources")
        gt = torch.utils.data.DataLoader(
            GroundTruthDataset(
                files_paths=[
                    os.path.join(resources_root, "test_midi1/piano_midi.mid"),
                    os.path.join(resources_root, "test_midi2/piano_midi.mid"),
                    os.path.join(resources_root, "test_midi3/piano_midi.mid"),
                ],
                condition_token="Hard Bop"
            ),
            batch_size=1,
            shuffle=True,
            drop_last=False
        )
        # Should have at least one item
        self.assertGreater(len(gt.dataset), 0)
        for b in gt:
            # Clamp features should just be a 768 dim tensor
            self.assertTrue(b.size(1) == 768)


class ReinforceGenerateTest(unittest.TestCase):
    cfg_parsed = training.parse_config_yaml(GENERATIVE_MODEL_CFG)
    cfg_parsed["reinforce_cfg"]["generated_sequence_length"] = 10  # short sequences for speed
    RG = ReinforceGenerateModule(**cfg_parsed)

    def test_condition_token_loader(self):
        # Dataset should contain 25 pianists, 20 genres
        self.assertTrue(len(self.RG.train_loader.dataset) == 45)
        # Every batch should have two elements: a condition token and BOS
        for i in self.RG.train_loader:
            self.assertTrue(i["condition_ids"].size(0) == 2)
            self.assertTrue(i["condition_ids"][1] == self.RG.tokenizer["BOS_None"])

    def test_do_generation(self):
        condition_tokens = torch.tensor([self.RG.tokenizer["PIANIST_BillEvans"], self.RG.tokenizer["BOS_None"]])
        gen_toks, gen_score = self.RG.do_generation(condition_tokens)
        # Check generated tokens
        self.assertTrue(isinstance(gen_toks, torch.Tensor))
        self.assertTrue(gen_toks.size(1) == self.RG.generated_sequence_length)
        # Check generated score
        self.assertTrue(isinstance(gen_score, Score))
        self.assertTrue(gen_score.ticks_per_quarter == utils.TICKS_PER_QUARTER)
        self.assertTrue(gen_score.time_signatures[0].numerator == utils.TIME_SIGNATURE)
        self.assertTrue(gen_score.tempos[0].qpm == utils.TEMPO)


if __name__ == '__main__':
    utils.seed_everything(utils.SEED)
    unittest.main()
