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

    def __init__(self, field, one_hot=True):
         super(LabelExtractor, self).__init__(field)
         self.one_hot = one_hot
         self.vocabulary = None

    def extract(self, labels, vocabulary):
        self.vocabulary = vocabulary[self.field]

        if isinstance(labels, list):
            ret = []
            for label in labels:
                if self.one_hot:
                    l = (np.arange(len(self.vocabulary)) == self.vocabulary[label]).astype(np.float32)
                else:
                    l = self.vocabulary[label]
                ret.append(l)
        else:
            if self.one_hot:
                ret = (np.arange(len(self.vocabulary)) == self.vocabulary[labels]).astype(np.float32)
            else:
                ret = self.vocabulary[labels]
                
        return ret
