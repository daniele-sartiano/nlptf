 # -*- coding: utf-8 -*-

import numpy as np

class WordEmbedding(object):
    def __init__(self, vocabulary, vectors, pad='</s>', unk='<unk>', size=50):
        self.vocabulary = vocabulary
        self.vectors = vectors
        self.matrix = np.matrix(vectors, dtype=np.float32)
        self.pad = pad
        self.unk = unk

        self.size = len(self.vectors[0]) if self.vectors else size
        self.number = len(self.vocabulary) if self.vocabulary else 0

    @property
    def padding(self):
        return self.vocabulary[self.pad]

    def w2e(self, word):
        return self.matrix[self.vocabulary[word]] if word in self.vocabulary else self.padding

    def w2idx(self, word):
        return self.vocabulary[word] if word in self.vocabulary else self.padding

