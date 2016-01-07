# -*- coding: utf-8 -*-

from itertools import izip

class Trainer(object):
    
    def __init__(self, reader, classifier):
        self.reader = reader
        self.classifier = classifier
    

class NerTrainer(Trainer):
    def train(self):
        x, y = self.reader.read()
        for i, sentence in enumerate(x):
            for ii, token in enumerate(sentence):
                #TODO: manage the train
                print token, y[i+ii]
