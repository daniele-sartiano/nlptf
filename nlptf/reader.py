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


class TextReader(Reader):
    UNK = '<unk>'
    PAD = '<pad>'

    def __init__(self, input, format=None, separator='\t', vocabulary=None):
        '''Build a Text Reader.

        :param input: The input file.
        :param format: A dictionary that describe the format of the input.
        :param separator: The separator of the file.
        :param vocabular: Th vocabulary.
        :return: a list of X, y
        '''

        super(TextReader, self).__init__(input)
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
    


class LineReader(TextReader):

    def read(self):
        examples = []
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
                examples.append(token)

        if not self.vocabulary:
            for field in self.format['fields']:
                if field['name'] != 'LABEL':
                    vocabulary[field['position']][self.UNK] = len(vocabulary[field['position']])
                    vocabulary[field['position']][self.PAD] = len(vocabulary[field['position']])

            self.vocabulary = vocabulary

        # Mapping Examples
        X = []
        y = []
        for example in examples:
            v = {}
            for field in self.format['fields']:
                if field['name'] != 'LABEL':
                    v[field['position']] = example[field['position']]

            X.append(v)
            y.append(example[self.getPosition('LABEL')])
        return X, y
        

class SentenceReader(TextReader):

    FORMAT = {
        'fields': [
            {'position': 0, 'name': 'FORM', 'type': str},
            {'position': 1, 'name': 'LABEL', 'type': str}
        ]
    }
            

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
                v = {}
                for field in self.format['fields']:
                    if field['name'] != 'LABEL':
                        v[field['position']] = token[field['position']]

                tokens_x.append(v)
                tokens_y.append(token[self.getPosition('LABEL')])
            X.append(tokens_x)
            y.append(tokens_y)
        return X, y

            
class IOBReader(SentenceReader):
    FORMAT = {
        'fields': [
            {'position': 0, 'name': 'FORM', 'type': str},
            {'position': 1, 'name': 'POS', 'type': str},
            {'position': 2, 'name': 'LABEL', 'type': str}
        ]
    }

    def map2idx(examples, labels, extractors, labelExtractor, wordEmbeddings):
        X = []
        y = []
        for example, listLabels in zip(examples, labels):
            feats = []
            for extractor in extractors:
                feats.append(extractor.extract(example, self.vocabulary))
            y.append(labelExtractor.extract(listLabels, self.vocabulary))
            X.append(([wordEmbeddings.w2idx(t[self.getPosition('FORM')]) for t in example], [el for el in zip(*feats)]))
        return X, y


class WebContentReader(LineReader):
    FORMAT = {
        'fields': [
            {'position': 0, 'name': 'DOMAIN', 'type': str},
            {'position': 1, 'name': 'LABEL', 'type': int},
            {'position': 2, 'name': 'TEXT', 'type': str}
        ]
    }

    def map2idx(self, examples, labels, extractors, labelExtractor, wordEmbeddings):
        X = []
        y = []
        for example, listLabels in zip(examples, labels):
            # import sys
            # print >> sys.stderr, 'ex:', example[0]
            feats = []
            for extractor in extractors:
                feats.append(extractor.extract(example, self.vocabulary))

            y.append(labelExtractor.extract(listLabels, self.vocabulary))
            X.append(([wordEmbeddings.w2idx(token) for token in example[self.getPosition('TEXT')]], [el for el in zip(*feats)]))

        return X, y
