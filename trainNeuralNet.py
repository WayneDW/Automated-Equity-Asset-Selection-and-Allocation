import numpy as np
import csv
import re
import tensorflow as tf
from datetime import datetime
from sklearn import metrics
from learning import learning


# implementation of neural networks in tensorflow
class neuralNet(learning): # inherit functions from learning class
    def __init__(self, date_start, date_end, date_type, sortino_rate, testYear):
        learning.__init__(self, date_start, date_end, date_type, sortino_rate, testYear)
        # convert training labels to one-hot
        self.Y = self.dense_to_one_hot(self.y, 2)

    # convert class labels from scalars to one-hot vectors
    # 0 => [1 0]
    # 1 => [0 1]
    def dense_to_one_hot(self, labels_dense, num_classes):
        num_labels = np.shape(labels_dense)[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    def feedForwardNN(self):
        seed = 128
        learning_rate = 0.001
        input_num_units = len(self.X[0])
        hidden_num_units = 50
        output_num_units = 2

        rng = np.random.RandomState(seed)
        # build the entire computation graph
        sess = tf.InteractiveSession()
    
        # build the computation graph by creating nodes for the input and labels
        # x and y_ aren't specific values. Rather, they are each a placeholder
        # a value that we'll input when we ask TensorFlow to run a computation.

        # define placeholders
        x = tf.placeholder(tf.float32, [None, input_num_units])
        y = tf.placeholder(tf.float32, [None, output_num_units])

        weights = {
            'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units], seed=seed)),
            'output': tf.Variable(tf.random_normal([hidden_num_units, output_num_units], seed=seed))
        }

        biases = {
            'hidden': tf.Variable(tf.random_normal([hidden_num_units], seed=seed)),
            'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))
        }
        
        hidden_layer = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'])
        output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']

        # more stable function
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_layer, y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        init = tf.initialize_all_variables()

        sess.run(init)
        sess.run(optimizer, feed_dict={x: self.X, y: self.Y})
        # no need to do output_layer = tf.nn.softmax(output_layer)
        predicted = sess.run(tf.argmax(output_layer, 1), feed_dict={x: self.X_test})

        self.ptLocal(self.fout, "Classification report for classifier :\n%s", \
            (metrics.classification_report(self.expected, predicted)))

    def ConvNet(self):
        sess = tf.InteractiveSession()  

        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        # Convolution and Pooling
        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        convLayer = 128
        hiddenLayer = 1024

        x = tf.placeholder(tf.float32, shape=[None, 6 * 28])
        y_ = tf.placeholder(tf.float32, shape=[None, 2])

        # First Convolutional Layer
        W_conv = weight_variable([1,2, 1, convLayer])
        b_conv = bias_variable([convLayer])
        x_image = tf.reshape(x, [-1,6,28,1])
        h_conv = tf.nn.relu(conv2d(x_image, W_conv) + b_conv)
        h_pool = max_pool_2x2(h_conv)

        W_fc1 = weight_variable([3*14*convLayer, hiddenLayer])
        b_fc1 = bias_variable([hiddenLayer])

        h_pool_flat = tf.reshape(h_pool, [-1, 3*14*convLayer])
        h_fc = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc, keep_prob)

        W_fc2 = weight_variable([hiddenLayer, 2])
        b_fc2 = bias_variable([2])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
        optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        init = tf.initialize_all_variables()

        sess.run(init)
        sess.run(optimizer, feed_dict={x: self.X, y_: self.Y, keep_prob: 0.2})
        # no need to do output_layer = tf.nn.softmax(output_layer)
        predicted = sess.run(tf.argmax(y_conv, 1), feed_dict={x: self.X_test, keep_prob: 1.0})

        print "Label 1 ratio: ", round(sum(predicted) * 1.0 / len(predicted), 3)
        self.ptLocal(self.fout, "Classification report for classifier :\n%s", \
            (metrics.classification_report(self.expected, predicted)))






if __name__ == "__main__":
    date_start = "2000-01-01"
    date_end = "2016-12-31"
    date_type = "d" # daily data
    sortino_rate = .5 # set classification threshold
    testYear = 2016
    s = neuralNet(date_start, date_end, date_type, sortino_rate, testYear)
    #s.feedForwardNN()
    s.ConvNet()
