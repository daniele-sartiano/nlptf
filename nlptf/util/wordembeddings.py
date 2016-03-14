 # -*- coding: utf-8 -*-

import numpy as np

class WordEmbedding(object):
    def __init__(self, vocabulary, vectors):
        self.vocabulary = vocabulary
        self.vectors = vectors
        self.matrix = np.matrix(vectors)

    @property
    def size(self):
        return len(self.vectors[0]) if self.vectors else 0
    
    @property
    def number(self):
        return len(self.vectors) if self.vectors else 0

    @property
    def padding(self):
        return self.vocabulary['</s>']

    def w2e(self, word):
        return self.matrix[self.vocabulary[word]] if word in self.vocabulary else self.padding

    def w2idx(self, word):
        return self.vocabulary[word] if word in self.vocabulary else self.padding
