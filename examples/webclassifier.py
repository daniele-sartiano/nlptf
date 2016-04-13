#!/usr/bin/env python

import sys
import os

sys.path.append('../nlptf')

import argparse

from nlptf.reader import Word2VecReader, WebContentReader
from nlptf.models.estimators import WordEmbeddingsEstimator, ConvWordEmbeddingsEstimator, RNNWordEmbeddingsEstimator, MultiRNNWordEmbeddingsEstimator, WordEmbeddingsEstimatorNC
from nlptf.extractors import FieldExtractor, CapitalExtractor
from nlptf.classifier.classifier import WordEmbeddingsClassifier

import tensorflow as tf

ESTIMATORS = {
    'linear': WordEmbeddingsEstimatorNC,
    'conv': ConvWordEmbeddingsEstimator,
    'rnn': RNNWordEmbeddingsEstimator,
    'multirnn': MultiRNNWordEmbeddingsEstimator
}

OPTIMIZERS = {
    'adam': tf.train.AdamOptimizer,
    'grad': tf.train.GradientDescentOptimizer
}


def main():

    parser = argparse.ArgumentParser(description='Named Entity Recognition with TensorFlow')
    subparsers = parser.add_subparsers()
    parser_train = subparsers.add_parser('train')
    parser_train.set_defaults(which='train')
    parser_train.add_argument('-e', '--epochs', help='epochs number', type=int, required=True)
    parser_train.add_argument('-l', '--learning-rate', help='learning rate', type=float, required=True)
    parser_train.add_argument('-o', '--optimizer', help='optimizer', type=str, required=True, choices=OPTIMIZERS.keys())

    parser_tag = subparsers.add_parser('tag')
    parser_tag.set_defaults(which='tag')

    parser_collect_data = subparsers.add_parser('collect')
    parser_collect_data.set_defaults(which='collect')
    parser_collect_data.add_argument('-d', '--directory', help='directory', type=str, required=True)
    parser_collect_data.add_argument('-i', '--input-file', help='input file', type=str, required=False)

    # common arguments
    for p in (parser_train, parser_tag):
        p.add_argument('-m', '--model', help='model-file', type=str, required=True)
        p.add_argument('-r', '--reader-file', help='reader file', type=str, required=True)
        p.add_argument('-w', '--word-embeddings', help='word embeddings', type=str, required=False)
        p.add_argument('-et', '--word-embeddings-type', help='word embeddings type', type=str, required=False)
        p.add_argument('-i', '--input-file', help='input file', type=str, required=False)
        p.add_argument('-t', '--type', help='estimator type', type=str, required=True, choices=ESTIMATORS.keys())
        p.add_argument('-nl', '--num-layers', help='number layers for multi rnn estimator', type=int, required=False)
        p.add_argument('-f', '--feats-conf', help='add the feats in the conf number', type=int, required=False)
        

    args = parser.parse_args()
    infile = open(args.input_file) if args.input_file is not None else sys.stdin

    if args.which == 'collect':
        with infile as f:
            for line in f:
                domain, agro, categories = line.strip().split('\t')
                # TODO: skipping multi-categories
                if ',' in categories:
                    continue
                cfile = os.path.join(args.directory, domain[0], domain[1], domain[2], domain, 'content.txt')
                try:
                    content = open(cfile).read()
                except:
                    print >> sys.stderr, '%s not found in %s' % (domain, cfile)
                    continue
                words = ' '.join([word.strip() for word in content.split()])
                if words:
                    print '%s\t%s\t%s' % (domain, categories, words)
    elif args.which == 'train':        
        reader = WebContentReader(infile, separator='\t')

        params = {
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'name_model': args.model, 
            'word_embeddings_file': args.word_embeddings,
            'reader_file': args.reader_file,
            'optimizer': OPTIMIZERS[args.optimizer],
            'num_layers': args.num_layers
        }

        classifier = WordEmbeddingsClassifier(reader, [], ESTIMATORS[args.type], **params)
        classifier.train()

    elif args.which == 'tag':

        lines = sys.stdin.readlines()
        reader = reader = WebContentReader(lines, separator='\t')

        extractors = []
        
        params = {
            'name_model': args.model,
            'word_embeddings_file': args.word_embeddings,
            'reader_file': args.reader_file,
            'num_layers': args.num_layers
        }

        classifier = WordEmbeddingsClassifier(reader, extractors, ESTIMATORS[args.type], **params)

        predicted = classifier.predict()

        print >> sys.stderr, len(predicted), len(lines)
        labels_idx_rev = {v:k for k,v in reader.vocabulary[reader.getPosition('LABEL')].items()}

        i = 0
        for line in lines:
            line = line.strip()
            if line:
                print '%s\t%s' % (line.split()[0], labels_idx_rev[predicted[i]])
                i += 1
            else:
                print


        

if __name__ == '__main__':
    main()
