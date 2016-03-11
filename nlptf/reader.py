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
            vocabulary[word] = len(vector)-1

        self.wordembedding = WordEmbedding(vocabulary=vocabulary, vectors=vectors)
        return self.wordembedding


class SentenceReader(Reader):
    @staticmethod
    def extractWindow(sentence, window):
        lpadded = window/2 * [-1] + list(sentence) + window/2 * [-1]
        return [lpadded[i:i+window] for i in range(len(sentence))] 

    @staticmethod
    def batch(sentence, size):
        out  = [sentence[:i] for i in xrange(1, min(size,len(sentence)+1) )]
        out += [sentence[i-size:i] for i in xrange(size,len(sentence)+1) ]
        return out

    
            
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
                    vocabulary[field['position']]['UNK'] = len(vocabulary[field['position']])
                    vocabulary[field['position']]['<PAD>'] = len(vocabulary[field['position']])

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


    def read_old2(self):
        sentence = []
        sentences = []

        vocabulary = {field['name']: {} for field in self.format['fields']}
        gazetter = {}

        for line in self.input:
            line = line.strip()
            if line:
                elements = line.split(self.separator)
                n_elements = len(elements)
                token = {}
                for field in self.format['fields']:
                    value = field['type'](elements[field['position']])
                    token[field['name']] = value
                    if value not in vocabulary[field['name']]:
                        vocabulary[field['name']][value] = len(vocabulary[field['name']])

                gaz_t_pos = self.format['gazetter']['type']['position']
                gaz_t_type = self.format['gazetter']['type']['type']
                gaz_v_pos = self.format['gazetter']['value']['position']
                gaz_v_type = self.format['gazetter']['value']['type']
                if gaz_t_type(elements[gaz_t_pos]) not in gazetter:
                    gazetter[gaz_t_type(elements[gaz_t_pos])] = set()
                gazetter[gaz_t_type(elements[gaz_t_pos])].add(gaz_v_type(elements[gaz_v_pos]))

                sentence.append(token)
                
            else:
                sentences.append(sentence)
                sentence = []
        if sentence:
            sentences.append(sentence)
            sentence = []

        for k in gazetter:
            print k, gazetter[k]

        if not self.vocabulary:
            for field in self.format['fields']:
                if field['name'] != 'LABEL':
                    vocabulary[field['name']]['UNK'] = len(vocabulary[field['name']])
                    vocabulary[field['name']]['PAD'] = -1
            self.vocabulary = vocabulary
        if not self.gazetter:
            self.gazetter = gazetter #todo the reverse
            
        # Mapping Sentence
        X = []
        y = []
        for sentence in sentences:
            # TODO: using only form
            tokens_x = []
            tokens_y = []

            for token in sentence:
                form = self.vocabulary['FORM'][token['FORM']] if token['FORM'] in self.vocabulary['FORM'] else self.vocabulary['FORM']['UNK']
                form /= float(len(self.vocabulary['FORM'])-1)

                pos = self.vocabulary['POS'][token['POS']] if token['POS'] in self.vocabulary['POS'] else self.vocabulary['POS']['UNK']
                pos /= float(len(self.vocabulary['POS'])-1)

                
                
                tokens_x.append(np.array([form, pos]))
                tokens_y.append((np.arange(len(self.vocabulary['LABEL'])) == self.vocabulary['LABEL'][token['LABEL']]).astype(np.float32))
            
            #X.append(np.array(tokens_x, dtype=np.float32)/(len(self.vocabulary['FORM'])-1))
            X.append(np.array(tokens_x, dtype=np.float32))
            y.append(np.array(tokens_y, dtype=np.float32))

        return X, y

                
    def read_old(self):
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

        # mapping sentences  #TODO: generalize for n fields
        X = [[[self.vocabulary['FORM'][w['FORM']], self.vocabulary['POS'][w['POS']]] for w in sentence] for sentence in sentences]

        # mapping labels
        self.labels_idx = {l:i for i, l in enumerate(set(labels))}

        y = np.zeros((len(labels), len(self.labels_idx)))
        for i, label in enumerate(labels):
            y[i] = (np.arange(y.shape[1]) == self.labels_idx[label]).astype(np.float32)
        

        # len_labels_set = len(labels_set)

        # if not self.labels_idx:
        #     for i, l in enumerate(labels_set):
        #         self.labels_idx[l] = np.zeros(len_labels_set)
        #         self.labels_idx[l][i] = 1
        
        # y = np.zeros((len(labels), len_labels_set))

        # for i, label in enumerate(labels):
        #     y[i] = self.labels_idx[label]

        return [X, y]
