# -*- coding: utf-8 -*-

import cPickle as pickle

from nlptf.extractors import LabelExtractor
from nlptf.util.wordembeddings import WordEmbedding
from nlptf.reader import Word2VecReader
import numpy as np


class Classifier(object):

    def __init__(self, reader, extractors, estimator, epochs, learning_rate, window_size, name_model, reader_file, optimizer):
        self.reader = reader
        self.extractors = extractors
        self.estimator = estimator
        self.label_extractor = LabelExtractor(self.reader.getPosition('LABEL'))

        # estimator params
        self.epochs = epochs 
        self.learning_rate = learning_rate
        self.window_size = window_size
        self.name_model = name_model
        self.reader_file = reader_file
        self.optimizer_type = optimizer

    def predict(self):
        self.reader.load(pickle.load(open(self.reader_file)))
        sentences, _ = self.reader.read()

        X = []
        for sentence in sentences:
            feats = []
            for extractor in self.extractors:
                feats.append(extractor.extract(sentence, self.reader.vocabulary))
            X.append([el for el in zip(*feats)])

        self.estimator = self.estimator(
            epochs = self.epochs,
            num_labels = len(self.reader.vocabulary[self.reader.getPosition('LABEL')]),
            learning_rate = self.learning_rate,
            window_size = self.window_size,
            num_feats= len(self.extractors), 
            name_model = self.name_model
        )

        predicted = self.estimator.predict(X)
        return predicted

    
    def train(self):
        sentences, labels = self.reader.read()
        X = []
        y = []
        for sentence, listLabels in zip(sentences, labels):
            feats = []
            for extractor in self.extractors:
                feats.append(extractor.extract(sentence, self.reader.vocabulary))
            X.append([el for el in zip(*feats)])
            y.append(self.label_extractor.extract(listLabels, self.reader.vocabulary))
            
        pickle.dump(self.reader.dump(), open(self.reader_file, 'wb'))

        # splitting in train and dev
        train_dataset = X[int(len(X)*0.3):]
        train_labels = y[int(len(X)*0.3):]
        dev_dataset = X[0:int(len(X)*0.3)]
        dev_labels = y[0:int(len(X)*0.3)]
                
        self.estimator = self.estimator(
            epochs = self.epochs,
            num_labels = len(self.reader.vocabulary[self.reader.getPosition('LABEL')]),
            learning_rate = self.learning_rate,
            window_size = self.window_size,
            num_feats= len(self.extractors), 
            name_model = self.name_model
        )

        path = self.estimator.train(train_dataset, train_labels, dev_dataset, dev_labels)
        return path


class WordEmbeddingsClassifier(Classifier):
    def __init__(self, reader, extractors, estimator, name_model, window_size=None, reader_file='reader.pkl', word_embeddings_file=None, epochs=None, learning_rate=None, optimizer=None, num_layers=None):
        
        super(WordEmbeddingsClassifier, self).__init__(reader, extractors, estimator, epochs, learning_rate, window_size, name_model, reader_file, optimizer)

        self.num_layers = num_layers

        self.word_embeddings = None

        if word_embeddings_file is not None:
            self.word_embeddings = Word2VecReader(open(word_embeddings_file)).read()

    def train(self):
        examples, labels = self.reader.read()

        # if there are not wordembeddings assign the vocabulary to the word embeddings
        if self.word_embeddings is None:
            self.word_embeddings = WordEmbedding(self.reader.vocabulary[self.reader.getPosition('FORM')], [], self.reader.PAD, self.reader.UNK)

        X, y = self.reader.map2idx(examples, labels, self.extractors, self.label_extractor, self.word_embeddings)

        pickle.dump(self.reader.dump(), open(self.reader_file, 'wb'))

        # splitting in train and dev
        train_dataset = X[int(len(X)*0.3):]
        train_labels = y[int(len(X)*0.3):]
        
        dev_dataset = X[0:int(len(X)*0.3)]
        dev_labels = y[0:int(len(X)*0.3)]                

        params = {
            'epochs' : self.epochs,
            'num_labels' : len(self.reader.vocabulary[self.reader.getPosition('LABEL')]),
            'learning_rate' : self.learning_rate,
            'window_size' : self.window_size,
            'num_feats' : len(self.extractors), 
            'name_model' : self.name_model,
            'word_embeddings' : self.word_embeddings,
            'optimizer' : self.optimizer_type
        }
        
        # TODO: check if multi layer rnn
        if self.num_layers is not None:
            params['num_layers'] = self.num_layers

        self.estimator = self.estimator(**params)

        path = self.estimator.train(
            X= train_dataset,
            y= train_labels,
            dev_X= dev_dataset,
            dev_y= dev_labels
        )
        return path


    def predict(self):
        self.reader.load(pickle.load(open(self.reader_file)))
        examples, _ = self.reader.read()

        # if there are not wordembeddings assign the vocabulary to the word embeddings
        if self.word_embeddings is None:
            self.word_embeddings = WordEmbedding(self.reader.vocabulary[self.reader.getPosition('FORM')], [], self.reader.PAD, self.reader.UNK)

        X, _ = self.reader.map2idx(examples, [], self.extractors, self.label_extractor, self.word_embeddings)


        import sys
        print >> sys.stderr, len(X)

        
        params = {
            'epochs' : self.epochs,
            'num_labels' : len(self.reader.vocabulary[self.reader.getPosition('LABEL')]),
            'learning_rate' : self.learning_rate,
            'window_size' : self.window_size,
            'num_feats' : len(self.extractors), 
            'name_model' : self.name_model,
            'word_embeddings' : self.word_embeddings
        }

        # TODO: check if multi layer rnn
        if self.num_layers is not None:
            params['num_layers'] = self.num_layers
                
        self.estimator = self.estimator(**params)

        predicted = self.estimator.predict(X)
        return predicted
