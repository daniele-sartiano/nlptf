#!/usr/bin/env python

import sys

sys.path.append('../nlptf')

import argparse

from nlptf.reader import IOBReader, Word2VecReader
from nlptf.models.estimators import WordEmbeddingsEstimator
from nlptf.classifier.classifier import WordEmbeddingsClassifier



def main():
    parser = argparse.ArgumentParser(description='Named Entity Recognition with TensorFlow')
    subparsers = parser.add_subparsers()
    parser_train = subparsers.add_parser('train')
    parser_train.set_defaults(which='train')
    parser_train.add_argument('-e', '--epochs', help='epochs number', type=int, required=True)
    parser_train.add_argument('-l', '--learning-rate', help='learning rate', type=float, required=True)
    parser_train.add_argument('-wi', '--window', help='context window size', type=int, required=True)
    parser_train.add_argument('-m', '--model', help='model-file', type=str, required=True)
    parser_train.add_argument('-w', '--word-embeddings', help='word embeddings', type=str, required=False)
    parser_train.add_argument('-et', '--word-embeddings-type', help='word embeddings type', type=str, required=False)
    parser_train.add_argument('-i', '--input-file', help='input file', type=str, required=False)

    parser_tag = subparsers.add_parser('tag')
    parser_tag.set_defaults(which='tag')
    parser_tag.add_argument('-m', '--model', help='model-file', type=str, required=True)
    parser_tag.add_argument('-l', '--learning-rate', help='learning rate', type=float, required=True)
    parser_tag.add_argument('-i', '--input-file', help='input file', type=str, required=False)
    parser_tag.add_argument('-w', '--word-embeddings', help='word embeddings', type=str, required=False)
    parser_tag.add_argument('-et', '--word-embeddings-type', help='word embeddings type', type=str, required=False)
    parser_tag.add_argument('-wi', '--window', help='context window size', type=int, required=True)


    args = parser.parse_args()
    infile = args.input_file if args.input_file is not None else sys.stdin

    if args.which == 'train':
        # we can pass the input format as argument
        f = {
            'fields': [
                {'position': 0, 'name': 'FORM', 'type': str},
                {'position': 1, 'name': 'POS', 'type': str},
                {'position': 2, 'name': 'LABEL', 'type': str}
            ]
        }
        
        reader = IOBReader(infile, separator='\t', format=f)
        extractors = []

        params = {
            'epochs': args.epochs,
            'learning_rate': args.learning_rate, 
            'window_size': args.window,
            'name_model': args.model, 
            'word_embeddings_file': args.word_embeddings
        }

        classifier = WordEmbeddingsClassifier(reader, extractors, WordEmbeddingsEstimator, **params)
        classifier.train()

    elif args.which == 'tag':

        lines = sys.stdin.readlines()
        reader = IOBReader(lines)

        extractors = []
        params = {
            'window_size': args.window,
            'learning_rate': args.learning_rate,
            'name_model': args.model,
            'word_embeddings_file': args.word_embeddings
        }

        classifier = WordEmbeddingsClassifier(reader, extractors, WordEmbeddingsEstimator, **params)
        predicted = classifier.predict()
        labels_idx_rev = {v:k for k,v in reader.vocabulary[reader.getPosition('LABEL')].items()}

        i = 0
        for line in lines:
            line = line.strip()
            if line:
                print '%s\t%s\t%s' % (line.split()[0], line.split()[1], labels_idx_rev[predicted[i]])
                i += 1
            else:
                print


if __name__ == '__main__':
    main()
