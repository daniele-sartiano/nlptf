import sys
import os

sys.path.append('../nlptf')

import argparse

from nlptf.reader import Word2VecReader, WebContentReader
from nlptf.models.estimators import WordEmbeddingsEstimator, ConvWordEmbeddingsEstimator, RNNWordEmbeddingsEstimator, MultiRNNWordEmbeddingsEstimator, WordEmbeddingsEstimatorNC
from nlptf.extractors import FieldExtractor, CapitalExtractor, LabelExtractor
from nlptf.classifier.classifier import WordEmbeddingsClassifier

import tensorflow as tf
import tensorflow.contrib.learn as skflow

from sklearn import preprocessing, metrics
from sklearn.metrics import f1_score

import numpy as np

ESTIMATORS = {
    'conv': WordEmbeddingsEstimatorNC,
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

    parser_score = subparsers.add_parser('score')
    parser_score.set_defaults(which='score')
    parser_score.add_argument('-p', '--predicted', help='predicted file', type=str, required=True)
    parser_score.add_argument('-g', '--gold', help='gold file', type=str, required=True)

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
    try:
        infile = open(args.input_file) if args.input_file is not None else sys.stdin
    except:
        pass

    print args

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
        
        max_size = 50000
        word_embeddings = Word2VecReader(open(args.word_embeddings)).read()

        reader = WebContentReader(infile, separator='\t')
        examples, labels = reader.read()

        label_extractor = LabelExtractor(reader.getPosition('LABEL'), one_hot=False)
        X, y = reader.map2idx(examples, labels, [], label_extractor, word_embeddings)

        _X = []
        for x, _ in X: # we dont want feats from extractors
            if len(x) < max_size:
                x = np.lib.pad(x, (0, max_size - len(x)), 'constant', constant_values=0)
            else:
                x = x[:max_size]
            _X.append(x)

        X = np.array(_X)
        y = np.array(y)

        X = preprocessing.StandardScaler().fit_transform(X)
        
        classifier = skflow.TensorFlowDNNClassifier(hidden_units=[100, 200, 100], n_classes=len(label_extractor.vocabulary))

        #classifier = skflow.TensorFlowLinearRegressor()

        #classifier.fit(X, y, logdir='log')
        classifier.fit(X, y)

        
        #### test

        reader = WebContentReader(open('data/fine-it-test'), separator='\t')
        examples, labels = reader.read()

        label_extractor = LabelExtractor(reader.getPosition('LABEL'), one_hot=False)
        X, y = reader.map2idx(examples, labels, [], label_extractor, word_embeddings)

        _X = []
        for x, _ in X: # we dont want feats from extractors
            if len(x) < max_size:
                x = np.lib.pad(x, (0, max_size - len(x)), 'constant', constant_values=0)
            else:
                x = x[:max_size]
            _X.append(x)

        X = np.array(_X)
        y = np.array(y)

        score = metrics.accuracy_score(y, classifier.predict(X))
        print("Accuracy: %f" % score)

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

        print >> sys.stderr, 'l predicted', len(predicted), 'l lines', len(lines)
        labels_idx_rev = {v:k for k,v in reader.vocabulary[reader.getPosition('LABEL')].items()}

        i = 0
        for line in lines:
            line = line.strip()
            if line:
                print '%s\t%s' % (line.split()[0], labels_idx_rev[predicted[i]])
                i += 1
            else:
                print
    elif args.which == 'score':
        gold_dict = {}
        for line in open(args.gold):
            domain, label = line.strip().split('\t')[:2]
            gold_dict[domain] = label
        
        y_true = []
        y_pred = []
        for line in open(args.predicted):
            domain, label = line.strip().split('\t')[:2]
            y_pred.append(int(label))
            y_true.append(int(gold_dict[domain]))

        print f1_score(y_true, y_pred, average='macro') 
            
        


if __name__ == '__main__':
    main()
