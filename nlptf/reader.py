# -*- coding: utf-8 -*-

import numpy as np
import itertools

from collections import Counter


class Reader(object):
    """
    Abstract class for readers.
    """
    
    def __init__(self, input):
        self.input = input

    def read(self):
        raise NotImplementedError()


class IOBReader(Reader):

    FORMAT = [
        (0, 'FORM', str),
        (1, 'POS', str),
        (2, 'LABEL', str)
    ]
    
    def __init__(self, input, format=None, separator='\t'):
        super(IOBReader, self).__init__(input)
        self.format = format if format is not None else self.FORMAT
        self.separator = separator
        self.vocabulary = {}
        
    def read(self):
        sentences = []
        labels = []
        fields = {}

        sentence = []
        for line in self.input:
            elems = line.strip().split(self.separator)
            if len(elems) > 1:
                token = {}
                label = None
                for pos, field, type in self.format:
                    if field == 'LABEL':
                        label = type(elems[pos])
                    else:
                        token[field] = type(elems[pos])

                        try:
                            fields[field].append(type(elems[pos]))
                        except: #just first time
                            fields[field] = []
                            fields[field].append(type(elems[pos]))

                sentence.append(token)
                labels.append(label)
            else:
                sentences.append(sentence)
                sentence = []
        if sentence:
            sentences.append(sentence)

        self.vocabulary = {}

        for _, field, _ in self.format:
            if field == 'LABEL':
                continue
            self.vocabulary[field] = {w[0]:i for i, w in enumerate(Counter(itertools.chain(fields[field])).most_common())}
        
        # mapping sentences
        x  = np.array([[[self.vocabulary['FORM'][w['FORM']], self.vocabulary['POS'][w['POS']]] for w in sentence] for sentence in sentences])
        
        # mapping labels
        labels_set = set(labels)
        len_labels_set = len(labels_set)
        labels_idx = {}
        
        for i, l in enumerate(labels_set):
            labels_idx[l] = np.zeros(len_labels_set)
            labels_idx[l][i] = 1

        y = np.zeros((len(labels), len_labels_set))

        for i, label in enumerate(labels):
            y[i] = labels_idx[label]

        return [x, y]
