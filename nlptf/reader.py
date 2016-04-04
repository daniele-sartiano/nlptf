# -*- coding: utf-8 -*-

import numpy as np
import itertools

from collections import Counter

from util.wordembeddings import WordEmbedding

class Reader(object):
    """
    Abstract class for readers.
    """

    def __init__(self, input):
        self.input = input

    def read(self):
        raise NotImplementedError()


class WordEmbeddingReader(Reader):
    def __init__(self, input):
        super(WordEmbeddingReader, self).__init__(input)
        self.wordembedding = None


class Word2VecReader(WordEmbeddingReader):
    def read(self):
        vocabulary = {}
        vectors = []

        header = True
        for line in self.input:
            line = line.strip()
            if header:
                header = False
                continue
            splitted = line.split()
            word, vector = splitted[0], [float(n) for n in splitted[1:]]

            vectors.append(vector)
            vocabulary[word] = len(vectors)-1

        self.wordembedding = WordEmbedding(vocabulary=vocabulary, vectors=vectors)
        return self.wordembedding


class SentenceReader(Reader):
    pass
            
class IOBReader(SentenceReader):
    FORMAT = {
        'fields': [
            {'position': 0, 'name': 'FORM', 'type': str},
            {'position': 1, 'name': 'POS', 'type': str},
            {'position': 2, 'name': 'LABEL', 'type': str}
        ],
        'gazetter': {
            'value': {'position': 0, 'name': 'FORM', 'type': str}, 
            'type': {'position': 2, 'name': 'LABEL', 'type': str}
        }
    }


    UNK = '<unk>'
    PAD = '<pad>'


    def __init__(self, input, format=None, separator='\t', vocabulary=None, gazetter=None):
        '''Construct an IOB Reader.

        :param input: The input file.
        :param format: A dictionary that describe the format of the input.
        :param separator: The separator of the file.
        :param vocabular: Th vocabulary.
        :return: a list of X, y
        '''
        super(IOBReader, self).__init__(input)
        self.format = format if format is not None else self.FORMAT

        self.field2pos = {}
        for field in self.format['fields']:
            self.field2pos[field['name']] = field['position']

        self.separator = separator
        self.vocabulary = {} if vocabulary is None else vocabulary


    def dump(self):
        return [self.format, self.separator, self.vocabulary]

    
    def load(self, params):
        self.format, self.separator, self.vocabulary = params


    def getPosition(self, fieldName):
        return self.field2pos[fieldName]
            

    def read(self):
        sentence = []
        sentences = []
        vocabulary = {field['position']: {} for field in self.format['fields']}

        for line in self.input:
            line = line.strip()
            if line:
                elements = line.split(self.separator)
                n_elements = len(elements)
                token = {}
                for field in self.format['fields']:
                    value = field['type'](elements[field['position']])
                    token[field['position']] = value
                    if value not in vocabulary[field['position']]:
                        vocabulary[field['position']][value] = len(vocabulary[field['position']])

                sentence.append(token)
                
            else:
                sentences.append(sentence)
                sentence = []

        if sentence:
            sentences.append(sentence)
            sentence = []

        if not self.vocabulary:
            for field in self.format['fields']:
                if field['name'] != 'LABEL':
                    vocabulary[field['position']][self.UNK] = len(vocabulary[field['position']])
                    vocabulary[field['position']][self.PAD] = len(vocabulary[field['position']])

            self.vocabulary = vocabulary

        # Mapping Sentence
        X = []
        y = []
        for sentence in sentences:
            tokens_x = []
            tokens_y = []
            for token in sentence:
                field_sorted = sorted(self.format['fields'], key=lambda x: x['position'])
                v = [token[f['position']] for f in field_sorted if f['name'] != 'LABEL']

                tokens_x.append(v)
                tokens_y.append(token[self.getPosition('LABEL')])
            X.append(tokens_x)
            y.append(tokens_y)
        return X, y
