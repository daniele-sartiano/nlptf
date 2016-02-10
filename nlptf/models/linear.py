# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

# def softmax_classifier(tensor_in, labels, weights, biases, name=None):
#     """Returns prediction and loss for softmax classifier."""
#     with tf.op_scope([tensor_in, labels], name, "softmax_classifier"):
#         logits = tf.nn.xw_plus_b(tensor_in, weights, biases) # Wx + b

#         xent = tf.nn.softmax_cross_entropy_with_logits(logits,
#                                                        labels,
#                                                        name="xent_raw")
#         loss = tf.reduce_mean(xent, name="xent")
#         predictions = tf.nn.softmax(logits, name=name)
#         return predictions, loss

# def logistic_regression(X, y):
#     """Creates logistic regression TensorFlow subgraph.

#     Args:
#         X: tensor or placeholder for input features,
#            shape should be [batch_size, n_features].
#         y: tensor or placeholder for target,
#            shape should be [batch_size, n_classes].

#     Returns:
#         Predictions and loss tensors.
#     """
#     with tf.variable_scope('logistic_regression'):
#         weights = tf.get_variable('weights', [X.get_shape()[1],
#                                               y.get_shape()[-1]])
#         bias = tf.get_variable('bias', [y.get_shape()[-1]])
#         return softmax_classifier(X, y, weights, bias)


class Classifier(object):
    
    @staticmethod
    def accuracy(predictions, labels):
        #print 'predictions', predictions
        #print 'labels', labels
        #print 'np.argmax(predictions, 1)', np.argmax(predictions, 1)
        #print 'np.argmax(labels, 1)', np.argmax(labels, 1)
        #print 'predictions.shape[0]', predictions.shape[0]
        #print '---'
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])


class LinearClassifier(Classifier):

    def __init__(self):
        #self.saver = tf.train.Saver(tf.all_variables())

        # Init variables
        self.num_steps = 100001 #TODO: param
        self.batch_size = 64 #TODO: param
        self.num_feats = 2 #TODO: dynamic
        self.num_labels = 3 #TODO: dynamic

        # define the graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Input data.
            # self.X = tf.placeholder(tf.float32, shape=(1, 2), name='trainset')
            # self.y = tf.placeholder(tf.float32, shape=(1, 3), name='labels')

            self.X = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_feats), name='trainset')
            self.y = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_labels), name='labels')
            self.dev_X = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_feats), name='devset')
            self.predictions, self.loss = self.logistic_regression(self.X, self.y)
            # Optimizer.
            self.optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(self.loss)

            #self.saver = tf.train.Saver(tf.all_variables())


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


    def save(self):
        #path = self.saver.save(self.session, "model.ckpt")
        #print 'Saved on %s' % path
        pass


    def train(self, X, y, dev_X, dev_y):

        row_sums = X.sum(axis=1)
        new_matrix = X / row_sums[:, np.newaxis] 
        X = new_matrix

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
                    
                    #print self.accuracy(self.dev_prediction.eval({self.dev_X: dev_X}), dev_y)
        #print self.model[0], self.model[1]
            
    
    def predict(self, X):
        pass


