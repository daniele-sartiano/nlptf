# -*- coding: utf-8 -*-

import unittest
import sys

from reader import IOBReader
from trainer import Trainer

class TestReader(unittest.TestCase):
    
    def test_reader(self):
        pass
        #i = sys.stdin.readlines()
        #reader = IOBReader(i)
        #n_lines = 0
        # for labels, sentence in reader.read():
        #     for label, token in zip(labels, sentence):
        #         n_lines += 1
        
        # c = len([el for el in i if el != '\n'])
        
        # self.assertEqual(c, n_lines)


class TestTrainer(unittest.TestCase):
    def test_trainer(self):
        reader = IOBReader(sys.stdin)
        trainer = Trainer(reader, None)
        trainer.train()
