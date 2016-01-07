# -*- coding: utf-8 -*-

import unittest
import sys
import os

from nlptf.reader import IOBReader
from nlptf.trainer import NerTrainer

class TestReader(unittest.TestCase):

    def test_reader(self):
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_reader.iob')
        reader = IOBReader(open(filename))
        sentences, labels = reader.read()
        self.assertEqual(len([token for sentence in sentences for token in sentence]), len(labels))


class TestTrainer(unittest.TestCase):
    def test_trainer(self):
        reader = IOBReader(sys.stdin)
        trainer = NerTrainer(reader, None)
        trainer.train()
