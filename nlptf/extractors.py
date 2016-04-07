# -*- coding: utf-8 -*-

import numpy as np


class Extractor(object):
    pass


class FieldExtractor(Extractor):
    def __init__(self, field):
        self.field = field
        
    def extract(self, sentence, vocabulary):
        vocabulary = vocabulary[self.field]
        ret = []
        for token in sentence:
            try:
                feat = vocabulary[token[self.field]]
            except Exception as e:
                feat = vocabulary['<unk>']
            feat /= float(len(vocabulary)-1) # normalize
            ret.append(feat)
        return ret


class CapitalExtractor(FieldExtractor):        
    def __init__(self, field):
        super(CapitalExtractor, self).__init__(field)
        self.vocabulary = {
            False: 0,
            True: 1
        }


    def extract(self, sentence, _):
        vocabulary = self.vocabulary
        ret = []
        for token in sentence:
            try:
                feat = vocabulary[token[self.field][0].isupper()]
            except Exception as e:
                feat = vocabulary['<unk>']
            feat /= float(len(vocabulary)-1) # normalize
            ret.append(feat)
        return ret


class LabelExtractor(FieldExtractor):
    def extract(self, labels, vocabulary):
        vocabulary = vocabulary[self.field]
        if isinstance(labels, list):
            ret = []
            for label in labels:
                ret.append((np.arange(len(vocabulary)) == vocabulary[label]).astype(np.float32))
        else:
            ret = (np.arange(len(vocabulary)) == vocabulary[labels]).astype(np.float32)
        return ret
