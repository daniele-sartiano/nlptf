# -*- coding: utf-8 -*-

import unittest
import sys
import os

from nlptf.reader import IOBReader
from nlptf.trainer import NerTrainer
from nlptf.models.linear import LinearClassifier

class TestReader(unittest.TestCase):

    def test_reader(self):
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_reader.iob')
        reader = IOBReader(open(filename))
        sentences, labels = reader.read()
        self.assertEqual(len([token for sentence in sentences for token in sentence]), len(labels))


class TestLinear(unittest.TestCase):

    def test_train(self):
        reader = IOBReader(sys.stdin)
        X, y = reader.read()
        dataset = []
        labels = []

        for i, sentence in enumerate(X):
            for ii, token in enumerate(sentence):
                dataset.append(token)
                labels.append(y[i+ii])


        # splitting in train and dev
        train_dataset = dataset[int(len(dataset)*0.3):]
        train_labels = labels[int(len(labels)*0.3):]
        dev_dataset = dataset[0:int(len(dataset)*0.3)]
        dev_labels = labels[0:int(len(dataset)*0.3)]
        
        c = LinearClassifier()
        path = c.train(train_dataset, train_labels,
                       dev_dataset, dev_labels)


    # def test_predict(self):
    #     reader = IOBReader(sys.stdin)
    #     trainer = NerTrainer(reader, LinearClassifier())
    #     trainer.train()
