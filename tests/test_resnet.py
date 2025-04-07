#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for resnet50 performer identification module"""

import os
import unittest

import torch

from jazz_style_conditioned_generation import utils
from jazz_style_conditioned_generation.encoders import ResNet50, load_performer_identifier


def handle_cuda_exceptions(f):
    """Skips a test when we get a CUDA out-of-memory error, allowing tests to run parallel with training runs."""

    def wrapper(*args, **kw):
        try:
            return f(*args, **kw)
        except torch.cuda.OutOfMemoryError:
            unittest.skip("Ignoring CUDA out of memory error!")

    return wrapper


class ResNet50Test(unittest.TestCase):
    def setUp(self):
        try:
            self.MODEL = ResNet50(num_classes=20).to(utils.DEVICE)
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            raise unittest.SkipTest("Ignoring CUDA out of memory error!")

    @handle_cuda_exceptions
    def test_resnet(self):
        self.assertEquals(self.MODEL.fc.out_features, 20)
        self.assertIsInstance(self.MODEL.layer1[0].bn1, torch.nn.BatchNorm2d)
        # Test forward
        x = torch.rand(4, 1, 88, 3000).to(utils.DEVICE)
        expected = (4, 20)
        actual = self.MODEL(x)
        self.assertEqual(actual.size(), expected)

    @handle_cuda_exceptions
    def test_forward_features(self):
        x = torch.rand(4, 1, 88, 3000).to(utils.DEVICE)  # fake input: batch = 4
        actual = self.MODEL.forward_features(x).size()
        self.assertEqual(actual, (4, 2048))

    @handle_cuda_exceptions
    @unittest.skipIf(os.getenv("REMOTE") == "true", "Skipping test on GitHub Actions")
    def test_load_checkpoint(self):
        """Test that we can load the performer identification model weights correctly"""
        md_loaded = load_performer_identifier()
        self.assertEqual(md_loaded.fc.out_features, 20)  # should have twenty classes


if __name__ == '__main__':
    utils.seed_everything(utils.SEED)
    unittest.main()
