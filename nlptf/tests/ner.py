# -*- coding: utf-8 -*-

import unittest
import sys
import os

import numpy as np
import cPickle as pickle

from nlptf.reader import IOBReader
from nlptf.models.estimators import LinearEstimator
from nlptf.classifier.classifier import Classifier
from nlptf.extractors import FieldExtractor, CapitalExtractor


class TestReader(unittest.TestCase):

    def test_reader(self):
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_reader.iob')
        reader = IOBReader(open(filename))

        reader = IOBReader(sys.stdin)
        
        sentences, labels = reader.read()
        self.assertEqual(len(sentences), len(labels))


class TestLinear(unittest.TestCase):

    def test_train(self):
        f = {
            'fields': [
                {'position': 0, 'name': 'FORM', 'type': str},
                {'position': 1, 'name': 'POS', 'type': str},
                {'position': 2, 'name': 'LABEL', 'type': str}
            ]
        }
        
        reader = IOBReader(sys.stdin, separator='\t', format=f)
        extractors = [
            FieldExtractor(reader.getPosition('FORM')), 
            FieldExtractor(reader.getPosition('POS')),
            CapitalExtractor(reader.getPosition('FORM')), 
        ]
        classifier = Classifier(reader, extractors, LinearEstimator)
        classifier.train()


    def test_predict(self):
        lines = sys.stdin.readlines()
        reader = IOBReader(lines)

        extractors = [
            FieldExtractor(reader.getPosition('FORM')), 
            FieldExtractor(reader.getPosition('POS')),
            CapitalExtractor(reader.getPosition('FORM')), 
        ]

        classifier = Classifier(reader, extractors, LinearEstimator)
        
        predicted = classifier.predict()
        #self.assertEqual(len(dataset), len(predicted))

        labels_idx_rev = {v:k for k,v in reader.vocabulary[reader.getPosition('LABEL')].items()}

        i = 0
        for line in lines:
            line = line.strip()
            if line:
                print '%s\t%s\t%s' % (line.split()[0], line.split()[1], labels_idx_rev[predicted[i]])
                i += 1
            else:
                print



    # def test_train2(self):
    #     f = {
    #         'fields': [
    #             {'position': 0, 'name': 'FORM', 'type': str},
    #             {'position': 1, 'name': 'POS', 'type': str},
    #             {'position': 2, 'name': 'LABEL', 'type': str}
    #         ]
    #     }

    #     reader = IOBReader(sys.stdin, separator='\t', format=f)
    #     X, y = reader.read()

    #     pickle.dump(reader.dump(), open('reader.pkl', 'wb'))
                
    #     dataset = []
    #     labels = []

    #     # splitting in train and dev
    #     train_dataset = X[int(len(X)*0.3):]
    #     train_labels = y[int(len(X)*0.3):]
    #     dev_dataset = X[0:int(len(X)*0.3)]
    #     dev_labels = y[0:int(len(X)*0.3)]
        
    #     c = LinearClassifier(num_feats=5*2, n_labels=len(reader.vocabulary['LABEL']))
    #     path = c.train(train_dataset, train_labels, dev_dataset, dev_labels)
    #     print 'saved in %s' % path

    # def test_predict(self):
    #     lines = sys.stdin.readlines()

    #     reader = IOBReader(lines)
    #     reader.load(pickle.load(open('reader.pkl')))
        
    #     X, _ = reader.read()

    #     c = LinearClassifier(num_feats=5*2, n_labels=len(reader.vocabulary['LABEL']))
    #     predicted = c.predict(X)
    #     #self.assertEqual(len(dataset), len(predicted))

    #     labels_idx_rev = {v:k for k,v in reader.vocabulary['LABEL'].items()}

    #     i = 0
    #     for line in lines:
    #         line = line.strip()
    #         if line:
    #             print '%s\t%s\t%s' % (line.split()[0], line.split()[1], labels_idx_rev[predicted[i]])
    #             i += 1
    #         else:
    #             print
        
