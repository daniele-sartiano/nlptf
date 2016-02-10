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
    FORMAT = {
        'fields': [
            (0, 'FORM', str),
            (1, 'POS', str)
        ],
        'label': (2, 'LABEL', str)
    }

    def __init__(self, input, format=None, separator='\t', vocabulary=None):
        '''Construct an IOB Reader.

        :param input: The input file.
        :param format: A dictionary that describe the format of the input.
        :param separator: The separator of the file.
        :param vocabular: Th vocabulary.
        :return: a list of X, y
        '''
        super(IOBReader, self).__init__(input)
        self.format = format if format is not None else self.FORMAT
        self.separator = separator
        self.vocabulary = {} if vocabulary is None else vocabulary

    def read(self):
        sentences = []
        labels = []
        sentence = []

        for line in self.input:
            elements = line.strip().split(self.separator)
            n_elements = len(elements)

            if n_elements >= len(self.format['fields']):
                token = {}
                for position, field, field_type in self.format['fields']:
                    token[field] = field_type(elements[position])

                label_position, _, label_type = self.format['label']
                # if there is also label, we build a list of labels
                if n_elements >= label_position:
                    labels.append(label_type(elements[label_position]))
                sentence.append(token)
            elif n_elements == 1:
                sentences.append(sentence)
                sentence = []
            else:  # wrong number of elements
                raise ValueError('Wrong Format')
        if sentence:
            sentences.append(sentence)

        if not self.vocabulary:
            for _, field, _ in self.format['fields']:
                self.vocabulary[field] = {w: i for i, w in enumerate(
                    set([token[field] for sentence in sentences for token in sentence]))}


        # for _, field, _ in self.format['fields']:
        #     self.vocabulary[field] = {w[0]: i for i, w in
        #                               enumerate(Counter(itertools.chain(fields[field])).most_common())}

        # mapping sentences  #TODO: generalize for n fields
        x = [[[self.vocabulary['FORM'][w['FORM']], self.vocabulary['POS'][w['POS']]] for w in sentence] for sentence in
             sentences]

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
