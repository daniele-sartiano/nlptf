# -*- coding: utf-8 -*-

from itertools import izip
import numpy as np

class Trainer(object):
    
    def __init__(self, reader, classifier):
        self.reader = reader
        self.classifier = classifier
    

class NerTrainer(Trainer):
    def train(self):
        X, y = self.reader.read()
        print 'total %s' % len(X)

        train_dataset = []
        train_labels = []
        dev_dataset = []
        dev_labels = []

        for i, sentence in enumerate(X):
            if 0 <= i < len(X)*0.1:
                dataset = dev_dataset
                labels = dev_labels
            else:
                dataset = train_dataset
                labels = train_labels
            for ii, token in enumerate(sentence):
                dataset.append(token)
                labels.append(y[i+ii])

                #self.classifier.train(np.asmatrix(token), np.asmatrix(y[i+ii]))
            #print i

        #print trainset_dataset
        self.classifier.train(np.array(train_dataset, dtype=float), np.array(train_labels),
                              np.array(dev_dataset, dtype=float), np.array(dev_labels)
        )

        # for i, sentence in enumerate(x):
        #     for ii, token in enumerate(sentence):
        #         #TODO: manage the train
        #         # print token, y[i+ii]
        #         # print np.array(token).shape, np.array(y[i+ii]).shape
        #         print np.asmatrix(token)
        #         self.classifier.train(np.asmatrix(token), np.asmatrix(y[i+ii]))
        #     print i
        #     if i > 400:
        #         break
        self.classifier.save()


    def predict(self):
        x, y = self.reader.read()
        for i, sentence in enumerate(x):
            for ii, token in enumerate(sentence):
                #TODO: manage the train
                # print token, y[i+ii]
                # print np.array(token).shape, np.array(y[i+ii]).shape
                print self.classifier.predict(np.asmatrix(token))
        
