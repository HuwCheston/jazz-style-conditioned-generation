#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for tokenizer"""

import unittest

from jazz_style_conditioned_generation.data.tokenizer import (
    check_loaded_config
)


class TokenizerTest(unittest.TestCase):
    def test_check_loaded_config(self):
        loaded_config = {
            "config_1": 0,  # matches
            "config_2": 1,  # doesn't match
            "config_3": 2,  # doesn't match
            "config_4": 3
        }
        desired_config = {
            "config_1": 0,
            "config_2": 3,
            "config_3": 5,
        }
        expected = [True, False, False]
        actual = list(check_loaded_config(loaded_config, desired_config))
        self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
