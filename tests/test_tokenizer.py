#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for tokenizers"""

import os
import unittest

from miditok import REMI, TSD, Structured, PerTok, MIDILike

from jazz_style_conditioned_generation import utils
from jazz_style_conditioned_generation.data.tokenizer import (
    get_tokenizer_class_from_string,
    TokTrainingIteratorAugmentation,
    load_or_train_tokenizer,
    DEFAULT_TOKENIZER_CLASS
)


class TokenizerTest(unittest.TestCase):
    def test_getter(self):
        types = [REMI, TSD, Structured, PerTok, MIDILike]
        names = ["remi", "tsd", "structured", "pertok", "midilike"]
        for ty, name in zip(types, names):
            actual = get_tokenizer_class_from_string(name)
            self.assertEqual(actual, ty)

    def test_training_iterator(self):
        token_factory = REMI()
        fps = [os.path.join(utils.get_project_root(), "tests/test_resources/test_midi1/piano_midi.mid")]
        tti = TokTrainingIteratorAugmentation(token_factory, fps)
        # We should only have one track, but we should be augmenting it multiple times
        self.assertEqual(len(tti.files_paths), 1)
        self.assertGreater(len(tti), 1)
        # We shouldn't be splitting IDs, so we should only get one item per track
        token_factory.config.encode_ids_split = "no"
        gotter = tti.__getitem__(0)
        self.assertEqual(len(gotter), 1)
        # Now we can try with splitting
        token_factory.config.encode_ids_split = "bar"
        gotter = tti.__getitem__(0)
        self.assertGreater(len(gotter), 1)

    def test_load_or_train_tokenizer(self):
        fps = [os.path.join(utils.get_project_root(), "tests/test_resources/test_midi1/piano_midi.mid")]
        # Test without training
        actual = load_or_train_tokenizer("tmp.json", dict(do_training=False), fps)
        self.assertTrue(isinstance(actual, get_tokenizer_class_from_string(DEFAULT_TOKENIZER_CLASS)))
        self.assertFalse(actual.is_trained)
        # We should dump the tokenizer to a json
        self.assertTrue(os.path.isfile("tmp.json"))
        os.remove("tmp.json")

        # Test with training
        actual = load_or_train_tokenizer("tmp.json", dict(do_training=True), fps)
        self.assertTrue(isinstance(actual, get_tokenizer_class_from_string(DEFAULT_TOKENIZER_CLASS)))
        self.assertTrue(actual.is_trained)
        # We should dump the tokenizer to a json
        self.assertTrue(os.path.isfile("tmp.json"))
        os.remove("tmp.json")


if __name__ == '__main__':
    unittest.main()
