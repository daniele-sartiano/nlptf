# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

class Classifier(object):

    @staticmethod
    def normalize(X):
        row_sums = X.sum(axis=1)
        new_matrix = X / row_sums[:, np.newaxis] 
        return new_matrix

    @staticmethod
    def accuracy(predictions, labels):
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])


class LinearClassifier(Classifier):

    def __init__(self):

        # Init variables
        self.num_steps = 100001 #TODO: param
        self.batch_size = 128 #TODO: param
        self.num_feats = 2 #TODO: dynamic
        self.num_labels = 3 #TODO: dynamic

        # define the graph
        self.graph = tf.Graph()
        with self.graph.as_default():

            self.X = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_feats), name='trainset')
            self.y = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_labels), name='labels')
            self.dev_X = tf.placeholder(tf.float32, name='devset')

            self.predictions, self.loss = self.logistic_regression(self.X, self.y)

            self.dev_prediction = tf.nn.softmax(tf.matmul(self.dev_X, self.weights) + self.bias)

            # Optimizer.
            self.optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(self.loss)

            self.saver = tf.train.Saver(tf.all_variables())


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
            return predictions, loss


    @property
    def weights_(self):
        """Returns weights of the linear classifier."""
        return self.get_tensor_value('logistic_regression/weights:0')

    @property
    def bias_(self):
        """Returns weights of the linear classifier."""
        return self.get_tensor_value('logistic_regression/bias:0')


    def save(self, session):
        path = self.saver.save(session, "model.ckpt")
        print 'Saved on %s' % path
        return path


    def train(self, X, y, dev_X, dev_y):

        X = np.array(X, dtype=float)
        y = np.array(y)
        dev_X = np.array(dev_X, dtype=float)
        dev_y = np.array(dev_y)

        X = self.normalize(X)
        dev_X = self.normalize(dev_X)
        
        with tf.Session(graph=self.graph) as session:
            session.run(tf.initialize_all_variables())
            for step in xrange(self.num_steps):
                offset = (step*self.batch_size) % (y.shape[0]- self.batch_size)
                batch_X = X[offset:(offset + self.batch_size), :]
                batch_y = y[offset:(offset + self.batch_size), :]

                feed_dict = {self.X: batch_X, self.y: batch_y}
                _, loss, predictions = session.run([self.optimizer, self.loss, self.predictions], feed_dict)
                if step % 500 == 0:
                    print 'step', step, 'loss %f' % loss
                    print 'accuracy %f' % self.accuracy(predictions, batch_y)
                    print self.accuracy(self.dev_prediction.eval({self.dev_X: dev_X}), dev_y)
            return self.save(session)
        
    
    def predict(self, X):
        pass


