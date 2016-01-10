# -*- coding: utf-8 -*-

import tensorflow as tf


def softmax_classifier(tensor_in, labels, weights, biases, name=None):
    """Returns prediction and loss for softmax classifier."""
    with tf.op_scope([tensor_in, labels], name, "softmax_classifier"):
        logits = tf.nn.xw_plus_b(tensor_in, weights, biases)
        xent = tf.nn.softmax_cross_entropy_with_logits(logits,
                                                       labels,
                                                       name="xent_raw")
        loss = tf.reduce_mean(xent, name="xent")
        predictions = tf.nn.softmax(logits, name=name)
        return predictions, loss

def logistic_regression(X, y):
    """Creates logistic regression TensorFlow subgraph.

    Args:
        X: tensor or placeholder for input features,
           shape should be [batch_size, n_features].
        y: tensor or placeholder for target,
           shape should be [batch_size, n_classes].

    Returns:
        Predictions and loss tensors.
    """
    with tf.variable_scope('logistic_regression'):
        tf.histogram_summary('logistic_regression.X', X)
        tf.histogram_summary('logistic_regression.y', y)
        weights = tf.get_variable('weights', [X.get_shape()[1],
                                              y.get_shape()[-1]])
        bias = tf.get_variable('bias', [y.get_shape()[-1]])
        tf.histogram_summary('logistic_regression.weights', weights)
        tf.histogram_summary('logistic_regression.bias', bias)
        return softmax_classifier(X, y, weights, bias)

class LinearClassifier(object):

    def __init__(self):
        graph = tf.Graph()
        with graph.as_default():
            # Input data.
            X = tf.placeholder(tf.int32)
            y = tf.placeholder(tf.int32)
            global_step = tf.Variable(0, name="global_step", trainable=False)

            with tf.Session(graph=graph) as session:
                tf.initialize_all_variables().run()
                print("Initialized")

                for step in xrange(10):
                    feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}
                    session.run()