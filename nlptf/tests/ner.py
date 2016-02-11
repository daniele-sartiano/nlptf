# -*- coding: utf-8 -*-

import unittest
import sys
import os

import numpy as np
import cPickle as pickle

from nlptf.reader import IOBReader
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

        pickle.dump([reader.vocabulary, reader.labels_idx], open('reader.pkl', 'wb'))
                
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
        

    def test_predict(self):
        vocab, labels_idx = pickle.load(open('reader.pkl'))
        
        lines = sys.stdin.readlines()

        reader = IOBReader(lines, vocabulary=vocab, labels_idx=labels_idx)
        X, _ = reader.read()

        dataset = []

        for i, sentence in enumerate(X):
            for ii, token in enumerate(sentence):
                dataset.append(token)
        
        c = LinearClassifier()
        predicted = c.predict(dataset)
        self.assertEqual(len(dataset), len(predicted))

        labels_list = [None]* len(labels_idx)
        for k, v in labels_idx.items():
            labels_list[np.argmax([v], 1)[0]] = k

        print labels_list

        i = 0
        for line in lines:
            line = line.strip()
            if line:                
                print '%s\t%s' % (line.split('\t')[0], labels_list[predicted[i]])
                i += 1
            else:
                print
        
