class WordEmbedding(object):
    def __init__(self, vocabulary, vectors):
        self.vocabulary = vocabulary
        self.vectors = vectors

    @property
    def size(self):
        return len(self.vectors[0]) if self.vectors else 0
    
    @property
    def number(self):
        return len(self.vectors) if self.vectors else 0
