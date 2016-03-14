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
    def extractWindow(sentence, window, PAD):
        lpadded = np.concatenate([window/2 * PAD, sentence, window/2 * PAD])
        if len(lpadded.shape) > 1:
            return [np.concatenate(lpadded[i:i+window]) for i in range(len(sentence))]
        else:
            return [lpadded[i:i+window] for i in range(len(sentence))]

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


class LinearEstimatorWE(Estimator):

    def __init__(self, epochs, num_labels, learning_rate, window_size, num_feats, name_model, word_embeddings):
        self.epochs = epochs 
        self.num_feats = num_feats
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.window_size = window_size
        self.name_model = name_model
        self.word_embeddings = word_embeddings

        print 'Word Embeddings (%s x %s)' % (self.word_embeddings.size, self.word_embeddings.number)

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


            # Embeddings Layer
            self.X_emb = tf.placeholder(tf.int32, shape=(None, self.window_size), name='trainset')
            self.embeddings = tf.get_variable("embedding", shape=(self.word_embeddings.number, self.word_embeddings.size))
            self.embedded_words = tf.nn.embedding_lookup(self.embeddings, self.X_emb)
            self.embedded_words_expanded = tf.expand_dims(self.embedded_words, -1)
            

            pool_size = 5
            sequence_length = self.X_emb.get_shape()[1]
            hidden_size = 128
            l2_reg_lambda=0.0
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

            # Keeping track of l2 regularization loss (optional)
            l2_loss = tf.constant(0.0)

            # Create a convolution + maxpool layer for the given pool size
            with tf.name_scope("conv-maxpool"):
                # Convolution Layer
                filter_shape = [pool_size, self.word_embeddings.size, 1, hidden_size]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[hidden_size]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_words_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                self.h_pool = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - pool_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")


            self.h_pool_flat = tf.reshape(self.h_pool, [-1, hidden_size])
            # Add dropout
            with tf.name_scope("dropout"):
                self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

            # Final (unnormalized) scores and predictions
            with tf.name_scope("output"):
                W = tf.Variable(tf.truncated_normal([hidden_size, self.num_labels], stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_labels]), name="b")
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
                self.we_predictions = tf.argmax(self.scores, 1, name="we_predictions")

            # CalculateMean cross-entropy loss
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.y)
                self.we_loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

            # Accuracy
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.we_predictions, tf.argmax(self.y, 1))
                self.we_accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="we_accuracy")


            #self.saver = tf.train.Saver(tf.all_variables())
            self.saver = tf.train.Saver()

        print 'Graph initialized'


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


    def train(self, X, y, dev_X, dev_y, train_emb, dev_emb):

        print 'Train started'

        with tf.Session(graph=self.graph) as session:
            session.run(tf.initialize_all_variables())
            session.run(self.embeddings.assign(self.word_embeddings.matrix))

            for step in xrange(self.epochs):
                for i in xrange(len(X)):
                    
                    PAD = [[-1]*len(X[i][0])]
                    cwords = self.extractWindow(X[i], self.window_size, PAD)
                    cembeddings = self.extractWindow(train_emb[i], self.window_size, [self.word_embeddings.padding])

                    feed_dict = {
                        self.X: cwords, 
                        self.y: y[i], 
                        self.X_emb: cembeddings,
                        self.dropout_keep_prob: 0.5
                    }
                    _, loss, predictions, we_loss, we_accuracy = session.run([self.optimizer, self.loss, self.predictions, self.we_loss, self.we_accuracy], feed_dict)
                    
                    if i % 1000 == 0:
                        print '\tstep', i, 'loss %f' % loss
                        print '\taccuracy %f' % self.accuracy(predictions, y[i])
                        print '\twe accuracy %f' % we_accuracy
                
                cwords_dev = []
                labels_dev = []
                cembeddings_dev = []
                for i in xrange(len(dev_X)):
                    PAD = [[-1]*len(dev_X[i][0])]
                    cwords_dev += self.extractWindow(dev_X[i], self.window_size, PAD)
                    cembeddings_dev += self.extractWindow(dev_emb[i], self.window_size, [self.word_embeddings.padding])
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
            

