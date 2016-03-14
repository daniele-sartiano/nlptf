# -*- coding: utf-8 -*-

import cPickle as pickle

from nlptf.extractors import LabelExtractor
from nlptf.util.wordembeddings import WordEmbedding
from nlptf.reader import Word2VecReader
import numpy as np


class Classifier(object):

    def __init__(self, reader, extractors, estimator, epochs, learning_rate, window_size, name_model):
        self.reader = reader
        self.extractors = extractors
        self.estimator = estimator
        self.labelExtractor = LabelExtractor(self.reader.getPosition('LABEL'))

        # estimator params
        self.epochs = epochs 
        self.learning_rate = learning_rate
        self.window_size = window_size
        self.name_model = name_model

    def predict(self):
        self.reader.load(pickle.load(open('reader.pkl')))
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
            y.append(self.labelExtractor.extract(listLabels, self.reader.vocabulary))
            
        pickle.dump(self.reader.dump(), open('reader.pkl', 'wb'))

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
    def __init__(self, reader, extractors, estimator, epochs, learning_rate, window_size, name_model, word_embeddings_file):
        
        super(WordEmbeddingsClassifier, self).__init__(reader, extractors, estimator, epochs, learning_rate, window_size, name_model)
        self.word_embeddings = Word2VecReader(open(word_embeddings_file)).read()
  

    def train(self):
        sentences, labels = self.reader.read()
        X = []
        y = []
        wordEmbeddings = []
        for sentence, listLabels in zip(sentences, labels):
            feats = []
            for extractor in self.extractors:
                feats.append(extractor.extract(sentence, self.reader.vocabulary))
            X.append([el for el in zip(*feats)])
            y.append(self.labelExtractor.extract(listLabels, self.reader.vocabulary))
            wordEmbeddings.append([self.word_embeddings.w2idx(t[self.reader.getPosition('FORM')]) for t in sentence])

        pickle.dump(self.reader.dump(), open('reader_we.pkl', 'wb'))

        # splitting in train and dev
        train_dataset = X[int(len(X)*0.3):]
        train_labels = y[int(len(X)*0.3):]
        train_embeddings = wordEmbeddings[int(len(X)*0.3):]
        
        dev_dataset = X[0:int(len(X)*0.3)]
        dev_labels = y[0:int(len(X)*0.3)]
        dev_embeddings = wordEmbeddings[0:int(len(X)*0.3)]
                
        self.estimator = self.estimator(
            epochs = self.epochs,
            num_labels = len(self.reader.vocabulary[self.reader.getPosition('LABEL')]),
            learning_rate = self.learning_rate,
            window_size = self.window_size,
            num_feats= len(self.extractors), 
            name_model = self.name_model,
            word_embeddings = self.word_embeddings
        )

        path = self.estimator.train(
            X= train_dataset,
            y= train_labels,
            dev_X= dev_dataset,
            dev_y= dev_labels,
            train_emb= train_embeddings, 
            dev_emb= dev_embeddings
        )
        return path
    