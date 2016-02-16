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


    @staticmethod
    def extractWindow(sentence, window):
        lpadded = window/2 * [-1] + list(sentence) + window/2 * [-1]
        return [lpadded[i:i+window] for i in range(len(sentence))]

    @staticmethod
    def batch(sentence, size):
        out  = [sentence[:i] for i in xrange(1, min(size,len(sentence)+1) )]
        out += [sentence[i-size:i] for i in xrange(size,len(sentence)+1) ]
        return out


class LinearClassifier(Classifier):

    def __init__(self, num_feats=2, n_labels=3):

        # TODO params
        self.epochs = 25 
        self.num_feats = num_feats
        self.num_labels = n_labels
        self.learning_rate = 0.01
        self.window = 5
        self.name_model = 'model.ckpt'

        #self.batch_size = 128

        # define the graph
        self.graph = tf.Graph()
        with self.graph.as_default():

            self.X = tf.placeholder(tf.float32, shape=(None, self.num_feats), name='trainset')
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


    def load(self, session):
        self.saver.restore(session, self.name_model)


    def train(self, X, y, dev_X, dev_y):

        with tf.Session(graph=self.graph) as session:
            session.run(tf.initialize_all_variables())
            for step in xrange(self.epochs):
                for i in xrange(len(X)):
                    #X[i] = self.normalize(np.array(X[i], dtype=np.float32))
                    cwords = self.extractWindow(X[i], self.window)
                    feed_dict = {self.X: cwords, self.y: y[i]}
                    _, loss, predictions = session.run([self.optimizer, self.loss, self.predictions], feed_dict)
                    
                    if i % 1000 == 0:
                        print '\tstep', i, 'loss %f' % loss
                        print '\taccuracy %f' % self.accuracy(predictions, y[i])
                
                cwords_dev = []
                labels_dev = []
                for i in xrange(len(dev_X)):
                    cwords_dev += self.extractWindow(dev_X[i], self.window)
                    labels_dev += list(dev_y[i])
                pred_y = self.dev_prediction.eval({self.dev_X: cwords_dev})
                
                print 'Epoch %s' % step, self.accuracy(pred_y, labels_dev)

            return self.save(session)


    def predict(self, X):
        # X = np.array(X, dtype=float)
        # X = self.normalize(X)
        cwords = []
        with tf.Session(graph=self.graph) as session:
            self.load(session)
            for i in xrange(len(X)):
                cwords += self.extractWindow(X[i], self.window)

            y_hat = session.run(self.predict_labels, {self.X: cwords})
            return y_hat
            
            # for step in xrange(self.epochs):
            #     offset = (step*self.batch_size) % (X.shape[0]- self.batch_size)
            #     batch_X = X[offset:(offset + self.batch_size), :]
            #     y_hat = session.run([self.predict_labels], {self.X: batch_X})
            #     yield y_hat
                        
                
                        


    def train_old(self, X, y, dev_X, dev_y):

        # X = np.array(X, dtype=float)
        # y = np.array(y)

        # dev_X = np.array(dev_X, dtype=float)
        # dev_y = np.array(dev_y)

        print X[0]
        X = self.normalize(X)
        print X[0]
        dev_X = self.normalize(dev_X)

        # # Debug data
        # def reformat(dataset, labels):
        #     dataset = dataset.reshape((-1, 28 * 28)).astype(np.float32)
        #     # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
        #     labels = (np.arange(10) == labels[:,None]).astype(np.float32)
        #     return dataset, labels

        # import cPickle as pickle
        # with open('/project/piqasso/Experiments/NegationAndSpeculation/scope/deepnl_experiments/tensorflow/notMNIST.pickle', 'rb') as f:
        #     save = pickle.load(f)
        #     train_dataset = save['train_dataset']
        #     train_labels = save['train_labels']
        #     valid_dataset = save['valid_dataset']
        #     valid_labels = save['valid_labels']
        #     test_dataset = save['test_dataset']
        #     test_labels = save['test_labels']
        #     del save  # hint to help gc free up memory

        # X, y = reformat(train_dataset, train_labels)
        # dev_X, dev_y = reformat(valid_dataset, valid_labels)

        # # End Debug data
        
        with tf.Session(graph=self.graph) as session:
            session.run(tf.initialize_all_variables())
            
            for step in xrange(self.epochs):
                offset = (step*self.batch_size) % (y.shape[0]- self.batch_size)
                batch_X = X[offset:(offset + self.batch_size), :]
                batch_y = y[offset:(offset + self.batch_size), :]
                feed_dict = {self.X: batch_X, self.y: batch_y}
                _, loss, predictions = session.run([self.optimizer, self.loss, self.predictions], feed_dict)
                if step % 50 == 0:
                    print 'step', step, 'loss %f' % loss
                    print 'accuracy %f' % self.accuracy(predictions, batch_y)
                    pred_y = self.dev_prediction.eval({self.dev_X: dev_X})
                    # print pred_y[0:10]
                    # print 'vs'
                    # print dev_y[0:10]
                    print(np.argmax(pred_y, 1)[0:10])
                    print(np.argmax(dev_y, 1)[0:10])
                    print dev_y[0:10]

                    print self.accuracy(pred_y, dev_y)
            
            return self.save(session)

