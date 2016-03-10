# -*- coding: utf-8 -*-

import cPickle as pickle

from nlptf.extractors import LabelExtractor
import numpy as np

def read_word_embeddings(word_embeddings_file):
    embeddings = []
    vocab = {}
    embeddings_len = -1
    embeddings_size = -1
    with open(word_embeddings_file) as f:
        header = True
        for line in f:
            line = line.strip()
            if header:
                header = False
                embeddings_len, embeddings_size = [int(el) for el in line.split()]
                continue
            splitted = line.split()
            word, vector = splitted[0], [float(n) for n in splitted[1:]]

            embeddings.append(vector)
            vocab[word] = len(embeddings)-1

    assert embeddings_len == len(embeddings)
    assert embeddings_size == len(embeddings[0])
    assert len(vocab) == embeddings_len

    return embeddings, vocab

class Classifier(object):
    def __init__(self, reader, extractors, estimator, epochs, learning_rate, window_size, name_model, word_embeddings_file=None):
        self.reader = reader
        self.extractors = extractors
        self.estimator = estimator
        self.labelExtractor = LabelExtractor(self.reader.getPosition('LABEL'))

        # estimator params
        self.epochs = epochs 
        self.learning_rate = learning_rate
        self.window_size = window_size
        self.name_model = name_model
    
        if word_embeddings_file is not None:
            self.vectors, self.vectors_vocab = Word2VecReader(word_embeddings_file).read()
            

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
