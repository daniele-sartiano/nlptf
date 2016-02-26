 # -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

class Estimator(object):

    @staticmethod
    def normalize(X):
        row_sums = X.sum(axis=1)
        new_matrix = X / row_sums[:, np.newaxis] 
        return new_matrix

    @staticmethod
    def accuracy(predictions, labels):
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])


    @staticmethod
    def extractWindow(sentence, window):
        n = len(sentence[0])
        lpadded = window/2 * [[-1]*n] + list(sentence) + window/2 * [[-1]*n]
        return [np.concatenate(lpadded[i:i+window]) for i in range(len(sentence))]

    @staticmethod
    def batch(sentence, size):
        out  = [sentence[:i] for i in xrange(1, min(size,len(sentence)+1) )]
        out += [sentence[i-size:i] for i in xrange(size,len(sentence)+1) ]
        return out


    def save(self, session):
        path = self.saver.save(session, self.name_model)
        print 'Saved on %s' % path
        return path


    def load(self, session):
        self.saver.restore(session, self.name_model)

        

class LinearEstimator(Estimator):

    
    def __init__(self, epochs, num_labels, learning_rate, window_size, num_feats, name_model):
        self.epochs = epochs 
        self.num_feats = num_feats
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.window_size = window_size
        self.name_model = name_model

        # define the graph
        self.graph = tf.Graph()
        with self.graph.as_default():

            self.X = tf.placeholder(tf.float32, shape=(None, self.num_feats*self.window_size), name='trainset')
            self.y = tf.placeholder(tf.float32, shape=(None, self.num_labels), name='labels')

            self.dev_X = tf.placeholder(tf.float32, name='devset')

            self.predictions, self.loss, self.logits = self.logistic_regression(self.X, self.y)

            self.predict_labels = tf.argmax(self.logits, 1, name="predictions")

            self.dev_prediction = tf.nn.softmax(tf.matmul(self.dev_X, self.weights) + self.bias)

            # Optimizer.
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

            #self.saver = tf.train.Saver(tf.all_variables())
            self.saver = tf.train.Saver()


    def logistic_regression(self, X, y):
        with tf.variable_scope('logistic_regression'):
            self.weights = tf.get_variable('weights', [X.get_shape()[1],
                                                       y.get_shape()[-1]])
            self.bias = tf.get_variable('bias', [y.get_shape()[-1]])
            return self.softmax_classifier(X, y)


    def softmax_classifier(self, tensor_in, labels):
        with tf.op_scope([tensor_in, labels], None, "softmax_classifier"):
            logits = tf.nn.xw_plus_b(tensor_in, self.weights, self.bias) # Wx + b
            
            xent = tf.nn.softmax_cross_entropy_with_logits(logits,
                                                           labels,
                                                           name="xent_raw")
            loss = tf.reduce_mean(xent, name="xent")
            predictions = tf.nn.softmax(logits)
            return predictions, loss, logits


    def train(self, X, y, dev_X, dev_y):

        with tf.Session(graph=self.graph) as session:
            session.run(tf.initialize_all_variables())
            for step in xrange(self.epochs):
                for i in xrange(len(X)):
                    cwords = self.extractWindow(X[i], self.window_size)
                    feed_dict = {self.X: cwords, self.y: y[i]}

                    _, loss, predictions = session.run([self.optimizer, self.loss, self.predictions], feed_dict)
                    
                    if i % 1000 == 0:
                        print '\tstep', i, 'loss %f' % loss
                        print '\taccuracy %f' % self.accuracy(predictions, y[i])
                
                cwords_dev = []
                labels_dev = []
                for i in xrange(len(dev_X)):
                    cwords_dev += self.extractWindow(dev_X[i], self.window_size)
                    labels_dev += list(dev_y[i])
                pred_y = self.dev_prediction.eval({self.dev_X: cwords_dev})
                
                print 'Epoch %s' % step, self.accuracy(pred_y, labels_dev)

            return self.save(session)


    def predict(self, X):
        cwords = []
        with tf.Session(graph=self.graph) as session:
            self.load(session)
            for i in xrange(len(X)):
                cwords += self.extractWindow(X[i], self.window_size)

            y_hat = session.run(self.predict_labels, {self.X: cwords})
            return y_hat
            


