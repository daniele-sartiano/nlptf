# -*- coding: utf-8 -*-

import cPickle as pickle

from nlptf.extractors import LabelExtractor

class Classifier(object):
    def __init__(self, reader, extractors, estimator):
        self.reader = reader
        self.extractors = extractors
        self.estimator = estimator
        self.labelExtractor = LabelExtractor(self.reader.getPosition('LABEL'))


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
            num_feats=len(self.extractors)*5,
            n_labels = len(self.reader.vocabulary[self.reader.getPosition('LABEL')])
        )
        predicted = self.estimator.predict(X)
        return predicted
    
    def train(self):
        sentences, labels = self.reader.read()
        X = []
        y = []
        for sentence, listLabels in zip(sentences, labels):
            feats = []
            #sentence_X = []
            for extractor in self.extractors:
                feats.append(extractor.extract(sentence, self.reader.vocabulary))
            # for el in zip(*feats):
            #     sentence_X.append(el)
            X.append([el for el in zip(*feats)])
            y.append(self.labelExtractor.extract(listLabels, self.reader.vocabulary))

        pickle.dump(self.reader.dump(), open('reader.pkl', 'wb'))

        # splitting in train and dev
        train_dataset = X[int(len(X)*0.3):]
        train_labels = y[int(len(X)*0.3):]
        dev_dataset = X[0:int(len(X)*0.3)]
        dev_labels = y[0:int(len(X)*0.3)]
        
        self.estimator = self.estimator(
            num_feats=len(self.extractors)*5,
            n_labels = len(self.reader.vocabulary[self.reader.getPosition('LABEL')])
        )
        path = self.estimator.train(train_dataset, train_labels, dev_dataset, dev_labels)

