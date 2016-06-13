# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from tensorflow.models.rnn import rnn, rnn_cell


class Estimator(object):

    
    def set_model(self):
        raise NotImplementedError()

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


    def logistic_regression(self, X, y):
        with tf.variable_scope('logistic_regression'):
            weights = tf.get_variable('weights', [X.get_shape()[1],
                                                       y.get_shape()[-1]])
            bias = tf.get_variable('bias', [y.get_shape()[-1]])

            logits = tf.nn.xw_plus_b(X, weights, bias) # Wx + b
            
            xent = tf.nn.softmax_cross_entropy_with_logits(logits,
                                                           y,
                                                           name="xent_raw")
            loss = tf.reduce_mean(xent, name="xent")
            predictions = tf.nn.softmax(logits)
            return predictions, loss, logits, weights, bias
        

class LinearEstimator(Estimator):
    
    def __init__(self, epochs, num_labels, learning_rate, window_size, num_feats, name_model):
        self.epochs = epochs 
        self.num_feats = num_feats
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.window_size = window_size
        self.name_model = name_model
        self.set_model()


    def set_model(self):
        # define the graph
        self.graph = tf.Graph()
        with self.graph.as_default():

            self.X = tf.placeholder(tf.float32, shape=(None, self.num_feats*self.window_size), name='trainset')
            self.y = tf.placeholder(tf.float32, shape=(None, self.num_labels), name='labels')

            self.dev_X = tf.placeholder(tf.float32, name='devset')

            self.predictions, self.loss, self.logits, self.weights, self.bias = self.logistic_regression(self.X, self.y)

            self.predict_labels = tf.argmax(self.logits, 1, name="predictions")

            self.dev_prediction = tf.nn.softmax(tf.matmul(self.dev_X, self.weights) + self.bias)

            # Optimizer.
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

            #self.saver = tf.train.Saver(tf.all_variables())
            self.saver = tf.train.Saver()


    def train(self, X, y, dev_X, dev_y):

        with tf.Session(graph=self.graph) as session:
            session.run(tf.initialize_all_variables())

            for step in xrange(self.epochs):
                for i in xrange(len(X)):
                    PAD = [-1]*len(X[i][0])
                    cwords = self.extractWindow(X[i], self.window_size, [PAD])
                    feed_dict = {self.X: cwords, self.y: y[i]}

                    _, loss, predictions = session.run([self.optimizer, self.loss, self.predictions], feed_dict)
                    
                    if i % 1000 == 0:
                        print '\tstep', i, 'loss %f' % loss
                        print '\taccuracy %f' % self.accuracy(predictions, y[i])
                
                cwords_dev = []
                labels_dev = []
                for i in xrange(len(dev_X)):
                    PAD = [-1]*len(dev_X[i][0])
                    cwords_dev += self.extractWindow(dev_X[i], self.window_size, [PAD])
                    labels_dev += list(dev_y[i])
                pred_y = self.dev_prediction.eval({self.dev_X: cwords_dev})
                
                print 'Epoch %s' % step, self.accuracy(pred_y, labels_dev)

            return self.save(session)


    def predict(self, X):
        cwords = []
        with tf.Session(graph=self.graph) as session:
            self.load(session)
            for i in xrange(len(X)):
                PAD = [-1]*len(X[i][0])
                cwords += self.extractWindow(X[i], self.window_size, [PAD])

            y_hat = session.run(self.predict_labels, {self.X: cwords})
            return y_hat            


class WordEmbeddingsEstimator(Estimator):

    def __init__(self, name_model, window_size, word_embeddings, epochs=None, num_labels=None, learning_rate=None, num_feats=0, optimizer=None):
        self.epochs = epochs 
        self.num_feats = num_feats
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.window_size = window_size
        self.name_model = name_model
        self.word_embeddings = word_embeddings
        self.optimizer_type = optimizer
        self.set_model()


    def init_vars(self):
        # Embeddings Layer
        self.X = tf.placeholder(tf.int32, shape=(None, self.window_size), name='trainset_embeddings')
        self.y = tf.placeholder(tf.float32, shape=(None, self.num_labels), name='labels')        
        self.features = tf.placeholder(tf.float32, shape=(None, self.num_feats*self.window_size), name='trainset_feats')

        if self.word_embeddings.vectors:
            self.embeddings = tf.Variable(self.word_embeddings.matrix, name="embedding")
        else:
            self.embeddings = tf.Variable(tf.random_uniform([self.word_embeddings.number, self.word_embeddings.size], -1.0, 1.0), name="embedding")
        

    def set_model(self):
        # define the graph
        self.graph = tf.Graph()
        with self.graph.as_default():

            self.init_vars()

            self.embedded_words = tf.nn.embedding_lookup(self.embeddings, self.X)

            self.embedded_words_reshaped = tf.reshape(self.embedded_words, (-1, self.window_size*self.word_embeddings.size))

            # Adding other features
            if self.num_feats:
                self.embedded_words_reshaped = tf.concat(1, [self.embedded_words_reshaped, self.features])

            # logistic regression
            self.predictions, self.loss, self.logits, self.weights, self.bias = self.logistic_regression(self.embedded_words_reshaped, self.y)

            self.predictions = tf.argmax(self.predictions, 1, name="predictions")
            
            # Accuracy
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
                
            # Optimizer.
            if self.optimizer_type is not None:
                self.optimizer = self.optimizer_type(self.learning_rate).minimize(self.loss)

            self.saver = tf.train.Saver()


    def train(self, X, y, dev_X, dev_y):

        BATCH_SIZE = 500
        with tf.Session(graph=self.graph) as session:

            session.run(tf.initialize_all_variables())

            for step in xrange(self.epochs):
                for i in xrange(len(X)):
                    cfeatures = []
                    embeddings, features = X[i]
                    cembeddings = self.extractWindow(embeddings, self.window_size, [self.word_embeddings.padding])
                    
                    # splitting in mini batch
                    cembeddings_batches = [cembeddings[index:index+BATCH_SIZE] for index in range(0, len(cembeddings), BATCH_SIZE)]
                    y_batches = [y[i][index:index+BATCH_SIZE] for index in range(0, len(y[i]), BATCH_SIZE)]

                    if self.num_feats:
                        PAD = [-1]*len(features[0])
                        cfeatures = self.extractWindow(features, self.window_size, [PAD])
                        cfeatures_batches = [cfeatures[index:index+BATCH_SIZE] for index in range(0, len(cfeatures), BATCH_SIZE)]

                    for index_batch in xrange(len(y_batches)):
                        
                        feed_dict = {
                            self.X: cembeddings_batches[index_batch],
                            self.y: y_batches[index_batch]
                        }

                        if self.num_feats:
                            feed_dict[self.features] = cfeatures_batches[index_batch]

                            _, loss, accuracy = session.run([self.optimizer, self.loss, self.accuracy], feed_dict)
                    
                    if i % 1000 == 0:
                        print '\tstep', i, 'loss %f' % loss
                        print '\taccuracy %f' % (accuracy * 100)
                
                # validation
                loss = 0
                accuracy = 0
                for i in xrange(len(dev_X)):
                    embeddings_dev, features_dev = dev_X[i]
                    cembeddings_dev = self.extractWindow(embeddings_dev, self.window_size, [self.word_embeddings.padding])
                    if self.num_feats:
                        PAD = [-1]*len(features_dev[0])
                        cfeatures_dev = self.extractWindow(features_dev, self.window_size, [PAD])

                    labels_dev = list(dev_y[i])

                    feed_dict = {
                        self.X: cembeddings_dev,
                        self.y: labels_dev
                    }

                    if self.num_feats:
                        feed_dict[self.features] = cfeatures_dev

                    l, a = session.run([self.loss, self.accuracy], feed_dict)
                    loss += l
                    accuracy += a
                print 'Epoch %s' % step, 'loss', loss, 'accuracy', (accuracy * 100)
                print 'Epoch %s' % (step/len(dev_X)), 'loss', (loss/len(dev_X)), 'accuracy', ((accuracy/len(dev_X)) * 100)
                    
            return self.save(session)


    def predict(self, X):
        y_hat = []
        with tf.Session(graph=self.graph) as session:
            self.load(session)
            for i in xrange(len(X)):
                embeddings, features = X[i]
                cembeddings = self.extractWindow(embeddings, self.window_size, [self.word_embeddings.padding])
                if self.num_feats:
                    PAD = [-1]*len(features[0])
                    cfeatures = self.extractWindow(features, self.window_size, [PAD])

                feed_dict = {
                    self.X: cembeddings
                }
                if self.num_feats:
                    feed_dict[self.features] = cfeatures
                y_hat.extend(session.run(self.predictions, feed_dict))
            return y_hat


class ConvWordEmbeddingsEstimator(WordEmbeddingsEstimator):

    def __init__(self, name_model, window_size, word_embeddings, epochs=None, num_labels=None, learning_rate=None, num_feats=None, optimizer=None):
        super(ConvWordEmbeddingsEstimator, self).__init__(name_model, window_size, word_embeddings, epochs, num_labels, learning_rate, num_feats, optimizer)

    def set_model(self):
        # define the graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            filter_size = 5
            num_filters = 128
            filter_shape = [filter_size, self.word_embeddings.size, 1, 128]
            
            self.init_vars()

            self.embedded_words = tf.nn.embedding_lookup(self.embeddings, self.X)
            self.embedded_words_expanded = tf.expand_dims(self.embedded_words, -1)

            with tf.name_scope("conv-layer1"):
                # Convolution Layer
                n_filters = 128
                pooling_window = 1
                pooling_strides = 1

                # [filter_height, filter_width, in_channels, out_channels]
                filters1 = tf.get_variable('filters1', 
                                           [self.window_size,
                                            self.word_embeddings.size, 
                                            self.embedded_words_expanded.get_shape()[-1], 
                                            n_filters], 
                                           tf.float32)

                conv1 = tf.nn.conv2d(
                    self.embedded_words_expanded,
                    filters1,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv1")

                bias1 = tf.get_variable('bias1', [1, 1, 1, n_filters], tf.float32)
                conv1 = conv1 + bias1

                # Apply nonlinearity
                conv1 = tf.nn.relu(conv1, name="relu")

                # Maxpooling over the outputs
                pool1 = tf.nn.max_pool(
                    conv1,
                    ksize=[1, pooling_window, 1, 1],
                    strides=[1, pooling_strides, 1, 1],
                    padding='SAME',
                    name="pool")
                pool1 = tf.transpose(pool1, [0, 1, 3, 2])

            with tf.name_scope("conv-layer2"):
                filters2 = tf.get_variable('filters2', [1,
                                                       n_filters, 
                                                       pool1.get_shape()[3],
                                                       n_filters], 
                                           tf.float32)

                conv2 = tf.nn.conv2d(
                    pool1,
                    filters2,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv2")
                bias2 = tf.get_variable('bias2', [1, 1, 1, n_filters], tf.float32)
                conv2 = conv2 + bias2
                
                pool2 = tf.squeeze(tf.reduce_max(conv2, 1), squeeze_dims=[1])

            # Final (unnormalized) scores and predictions
            with tf.name_scope("output"):
                W = tf.get_variable('W', [pool2.get_shape()[1], self.num_labels])
                b = tf.get_variable('b', [self.num_labels])
                self.scores = tf.nn.xw_plus_b(pool2, W, b, name="scores") #logits
                #self.predictions = tf.argmax(self.scores, 1, name="predictions")
                self.predictions = tf.argmax(tf.nn.softmax(self.scores), 1, name="predictions")

            # Calculate Mean cross-entropy loss
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.y) #xent
                self.loss = tf.reduce_mean(losses)

            # Accuracy
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

            if self.optimizer_type is not None:
                self.optimizer = self.optimizer_type(self.learning_rate).minimize(self.loss)

            self.saver = tf.train.Saver()


class RNNWordEmbeddingsEstimator(WordEmbeddingsEstimator):

    def __init__(self, name_model, window_size, word_embeddings, epochs=None, num_labels=None, learning_rate=None, num_feats=None, optimizer=None):
        super(RNNWordEmbeddingsEstimator, self).__init__(name_model, window_size, word_embeddings, epochs, num_labels, learning_rate, num_feats, optimizer)

    def set_model(self):
        # define the graph
        self.graph = tf.Graph()
        with self.graph.as_default():

            self.init_vars()
            
            self.embedded_words = tf.nn.embedding_lookup(self.embeddings, self.X)

            word_list =  [tf.squeeze(t, squeeze_dims=[1]) for t in tf.split(1, self.window_size, self.embedded_words)]
            cell = rnn_cell.GRUCell(self.word_embeddings.size)
            _, encoding = rnn.rnn(cell, word_list, dtype=tf.float32)

            self.predictions, self.loss, self.logits, self.weights, self.bias = self.logistic_regression(encoding, self.y)
            self.predictions = tf.argmax(self.predictions, 1)
            
            # Accuracy
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

            if self.optimizer_type is not None:
                self.optimizer = self.optimizer_type(self.learning_rate).minimize(self.loss)

            self.saver = tf.train.Saver()


class MultiRNNWordEmbeddingsEstimator(RNNWordEmbeddingsEstimator):

    def __init__(self, name_model, window_size, word_embeddings, epochs=None, num_labels=None, learning_rate=None, num_feats=None, optimizer=None, num_layers=3):
        self.num_layers = num_layers
        super(MultiRNNWordEmbeddingsEstimator, self).__init__(name_model, window_size, word_embeddings, epochs, num_labels, learning_rate, num_feats, optimizer)


    def set_model(self):

        # define the graph
        self.graph = tf.Graph()
        with self.graph.as_default():

            self.init_vars()
            
            self.embedded_words = tf.nn.embedding_lookup(self.embeddings, self.X)

            word_list =  [tf.squeeze(t, squeeze_dims=[1]) for t in tf.split(1, self.window_size, self.embedded_words)]
            grucell = rnn_cell.GRUCell(self.word_embeddings.size)
            cell = tf.nn.rnn_cell.MultiRNNCell([grucell] * self.num_layers)
            _, encoding = tf.nn.rnn(cell, word_list, dtype=tf.float32)

            self.predictions, self.loss, self.logits, self.weights, self.bias = self.logistic_regression(encoding, self.y)
            self.predictions = tf.argmax(self.predictions, 1)
            
            # Accuracy
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

            if self.optimizer_type is not None:
                self.optimizer = self.optimizer_type(self.learning_rate).minimize(self.loss)

            self.saver = tf.train.Saver()


class WordEmbeddingsEstimatorNC(Estimator):

    def __init__(self, name_model, window_size, word_embeddings, epochs=None, num_labels=None, learning_rate=None, num_feats=0, optimizer=None):
        self.epochs = epochs 
        self.num_feats = num_feats
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.batch_size = 1
        self.name_model = name_model
        self.word_embeddings = word_embeddings
        self.optimizer_type = optimizer
        self.max_size = 5000
        self.set_model()

    def init_vars(self):
        # Embeddings Layer
        self.X = tf.placeholder(tf.int32, shape=(None, self.max_size), name='trainset_embeddings')
        self.y = tf.placeholder(tf.float32, shape=(None, self.num_labels), name='labels')        

        self.embeddings = tf.Variable(tf.random_uniform([self.word_embeddings.number, self.word_embeddings.size], -1.0, 1.0), name="embedding")

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    def set_model(self):
        # define the graph
        self.graph = tf.Graph()
        with self.graph.as_default():

            self.init_vars()

            self.embedded_words = tf.nn.embedding_lookup(self.embeddings, self.X)

            #self.embedded_words_reshaped = tf.reshape(self.embedded_words, (-1, self.batch_size*self.word_embeddings.size))
            self.embedded_words_expanded = tf.expand_dims(self.embedded_words, -1)

            with tf.name_scope("conv-layer1"):
                # Convolution Layer
                #n_filters = 128
                n_filters = 64
                pooling_window = 1
                pooling_strides = 1
                filter_size = 5

                # [filter_height, filter_width, in_channels, out_channels]
                filter_shape = [filter_size, self.word_embeddings.size, 1, n_filters]
                filters1 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='filters1')

                # input --> [batch, in_height, in_width, in_channels]
                conv1 = tf.nn.conv2d(
                    self.embedded_words_expanded,
                    filters1,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv1")

                self.conv1before = conv1
                bias  = tf.Variable(tf.constant(0.1, shape=[n_filters]))
                self.conv1bias = tf.nn.bias_add(conv1, bias)

                # Apply nonlinearity
                self.conv1 = tf.nn.relu(self.conv1bias, name="relu")

                # Maxpooling over the outputs
                self.pool1 = tf.nn.max_pool(
                    self.conv1,
                    ksize=[1, self.max_size - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")

            self.h_pool = tf.concat(3, [self.pool1])
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, n_filters])
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

            # Final (unnormalized) scores and predictions
            with tf.name_scope("output"):
                W = tf.get_variable('W', [n_filters, self.num_labels], initializer=tf.contrib.layers.xavier_initializer())
                #b = tf.get_variable('b', [self.num_labels])
                b = tf.Variable(tf.constant(0.1, shape=[self.num_labels]), name="b")
                self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores") #logits
                self.predictions = tf.argmax(self.scores, 1, name="predictions")

            # Calculate Mean cross-entropy loss
            with tf.name_scope("loss"):
                self.losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.y) #xent
                self.loss = tf.reduce_mean(self.losses)

            # Accuracy
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

            # if self.optimizer_type is not None:
            #     self.optimizer = self.optimizer_type(self.learning_rate).minimize(self.loss)

            global_step = tf.Variable(0, name="global_step", trainable=False) 
            self.optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = self.optimizer.compute_gradients(self.loss)
            self.train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            self.saver = tf.train.Saver()


    def train(self, X, y, dev_X, dev_y):
        with tf.Session(graph=self.graph) as session:

            session.run(tf.initialize_all_variables())

            for step in xrange(self.epochs):
                for i in xrange(len(X)):
                    embeddings, features = X[i]
                    
                    if len(embeddings) < self.max_size:
                        example = [np.lib.pad(embeddings, (0, self.max_size - len(embeddings)), 'constant', constant_values=0)]
                    else:
                        example = [np.array(embeddings[:self.max_size])]

                    feed_dict = {
                        self.X: example,
                        self.y: [y[i]],
                        self.dropout_keep_prob: 0.5
                    }
                    
                    _, loss, accuracy, eee, emb_exp, conv1before, conv1bias, conv1, scores, losses, pool1 = session.run([self.train_op, self.loss, self.accuracy, self.embedded_words, self.embedded_words_expanded, self.conv1before, self.conv1bias, self.conv1, self.scores, self.losses, self.pool1], feed_dict)

                    if i % 100 == 0:
                        print '\tstep', i, 'loss %f' % loss

                        # print example
                        # print y[i]
                        # print '***'
                        # print 'embeddings', eee.shape, eee
                        # print 'embeddings exapanded', emb_exp.shape, emb_exp
                        # print 'conv1 before', conv1before.shape, conv1before
                        # print 'conv1 bias', conv1bias.shape, conv1bias
                        # print 'conv1', conv1.shape, conv1
                        # print 'scores', scores.shape, scores
                        # print 'losses', losses.shape, losses
                        # print 'pool1', pool1.shape, pool1
                

                loss = 0
                accuracy = 0
                for i in xrange(len(dev_X)):
                    emb_dev, feats_dev = dev_X[i]
                    if len(emb_dev) < self.max_size:
                        example = np.lib.pad(emb_dev, (0, self.max_size - len(emb_dev)), 'constant', constant_values=0)
                    else:
                        example = np.array(emb_dev[:self.max_size])
                    
                    feed_dict = {
                        self.X: [example],
                        self.y: [dev_y[i]],
                        self.dropout_keep_prob: 0.5
                    }

                    l, a = session.run([self.loss, self.accuracy], feed_dict)
                    loss += l
                    accuracy += a

                print 'Epoch %s' % step, 'loss', (loss/len(dev_X)), 'accuracy', ((accuracy/len(dev_X)) * 100)

            return self.save(session)


    def predict(self, X):

        import sys
        print >> sys.stderr, len(X)
                        
        with tf.Session(graph=self.graph) as session:
            self.load(session)
            y_hats = []
            for i in xrange(len(X)):
                emb, features = X[i]
                
                if len(emb) < self.max_size:
                    example = np.lib.pad(emb, (0, self.max_size - len(emb)), 'constant', constant_values=0)
                else:
                    example = emb[:self.max_size]

                feed_dict = {
                    self.X: [example],
                    self.dropout_keep_prob: 0.5
                }

                y_hat = session.run(self.predictions, feed_dict)
                y_hats.extend(y_hat)
            return y_hats
