# -*- coding: utf-8 -*-

import unittest
import sys
import os

import numpy as np
import cPickle as pickle

from nlptf.reader import IOBReader, Word2VecReader
from nlptf.models.estimators import LinearEstimator, WordEmbeddingsEstimator
from nlptf.classifier.classifier import Classifier, WordEmbeddingsClassifier
from nlptf.extractors import FieldExtractor, CapitalExtractor


class TestReader(unittest.TestCase):

    def test_reader(self):
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_reader.iob')
        reader = IOBReader(open(filename))

        reader = IOBReader(sys.stdin)
        
        sentences, labels = reader.read()
        self.assertEqual(len(sentences), len(labels))

    def test_we_reader(self):
        reader = Word2VecReader(sys.stdin)
        we = reader.read()
        assert we.number == len(we.vectors)
        assert we.size == len(we.vectors[0])
        assert len(we.vocabulary) == we.number


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

        params = {'epochs':25, 'learning_rate':0.01, 'window_size':3, 'name_model':'model.ckpt'}
        classifier = Classifier(reader, extractors, LinearEstimator, **params)
        classifier.train()


    def test_predict(self):
        lines = sys.stdin.readlines()
        reader = IOBReader(lines)

        extractors = [
            FieldExtractor(reader.getPosition('FORM')), 
            FieldExtractor(reader.getPosition('POS')),
            CapitalExtractor(reader.getPosition('FORM')), 
        ]

        params = {'epochs':25, 'learning_rate':0.01, 'window_size':3, 'name_model':'model.ckpt'}
        classifier = Classifier(reader, extractors, LinearEstimator, **params)
        
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


class TestWordEmbeddings(unittest.TestCase):

    def test_train(self):
        f = {
            'fields': [
                {'position': 0, 'name': 'FORM', 'type': str},
                {'position': 1, 'name': 'POS', 'type': str},
                {'position': 2, 'name': 'LABEL', 'type': str}
            ]
        }
        
        reader = IOBReader(sys.stdin, separator='\t', format=f)
        extractors = []

        params = {
            'epochs': 15,
            'learning_rate': 0.01, 
            'window_size': 5,
            'name_model': 'model_we.ckpt', 
            'word_embeddings_file': 'data/vectors.txt'
        }

        classifier = WordEmbeddingsClassifier(reader, extractors, WordEmbeddingsEstimator, **params)
        classifier.train()


    def test_predict(self):
        lines = sys.stdin.readlines()
        reader = IOBReader(lines)

        extractors = []

        params = {
            'epochs': 15,
            'learning_rate': 0.01, 
            'window_size': 5, 
            'name_model': 'model_we.ckpt',
            'word_embeddings_file': 'data/vectors.txt'
        }
        classifier = WordEmbeddingsClassifier(reader, extractors, WordEmbeddingsEstimator, **params)
        predicted = classifier.predict()
        labels_idx_rev = {v:k for k,v in reader.vocabulary[reader.getPosition('LABEL')].items()}

        i = 0
        for line in lines:
            line = line.strip()
            if line:
                print '%s\t%s\t%s' % (line.split()[0], line.split()[1], labels_idx_rev[predicted[i]])
                i += 1
            else:
                print
