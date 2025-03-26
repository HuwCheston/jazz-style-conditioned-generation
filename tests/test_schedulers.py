#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for LR schedulers in encoders/schedulers.py"""

import unittest

import torch

from jazz_style_conditioned_generation.encoders.scheduler import DummyScheduler, WarmupScheduler


class SchedulerTest(unittest.TestCase):
    def test_no_op_scheduler(self):
        # Define a simple model and optimizer
        model = torch.nn.Linear(in_features=1, out_features=32)
        optim = torch.optim.Adam(model.parameters(), lr=0.001)
        # Define our custom scheduler
        scheduler = DummyScheduler(optimizer=optim)
        # Iterate over some `epochs`
        for i in range(100):
            optim.zero_grad()
            # model would train here
            optim.step()
            # After stepping in the scheduler, the LR should not have changed from the initial value
            scheduler.step()
            for lr in scheduler.get_lr():
                self.assertEqual(lr, 0.001)
        # Check that we have a state dict
        self.assertTrue(hasattr(scheduler, 'state_dict'))

    def test_warmup_scheduler(self):
        # Define parameters for the test
        min_lr = 2e-6
        max_lr = 1e-2
        warmup_steps = 100
        gamma = 0.9999
        # Define a simple model and optimizer
        model = torch.nn.Linear(in_features=1, out_features=32)
        optim = torch.optim.Adam(model.parameters(), lr=min_lr)
        # Define our warmup scheduler
        warmup = WarmupScheduler(optim, max_lr=max_lr, warmup_steps=warmup_steps, gamma=gamma)
        all_lrs = []
        # Iterate over some "batches"
        for _ in range(1000):
            all_lrs.append(warmup.get_last_lr()[-1])
            optim.zero_grad()
            optim.step()
            warmup.step()
        # Test all LRs, should be within expected boundaries
        self.assertTrue(min(all_lrs) == min_lr)
        self.assertTrue(max(all_lrs) == max_lr)
        # Test first and last results for warmup period
        self.assertTrue(all_lrs[0] == min_lr)
        self.assertTrue(all_lrs[warmup_steps] == max_lr)


if __name__ == '__main__':
    unittest.main()
