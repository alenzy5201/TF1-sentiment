import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    CNN用于情感分析
    整个CNN架构包括词嵌入层,卷积层,max-pooling层,隐藏层,学习的嵌入层,和softmax层
    """
    def __init__(
      self, sequence_length, num_classes, embedding_table,
            filter_sizes, num_filters,  l2_reg_lambda=50):
            # Inputs
            self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
            #print(self.input_x.graph)
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

            #l2 regularization loss
            l2_loss = tf.constant(0.0)
            self.dropout_keep_prob = 0.8

            # Word2vec embedding
            with  tf.name_scope("embedding"):
                #self.W = tf.Variable(embedding_table,name="W2vec", trainable=False)
                self.W = tf.Variable(embedding_table, name="W2vec", trainable=False)
                #print(self.W.graph)
                self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

            # Conv and max-pooling
            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size,300, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.tanh(tf.nn.bias_add(conv, b), name="tanh")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)

            # Flatten
            num_filters_total = num_filters * len(filter_sizes)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(  self.h_pool, [-1, num_filters_total])

            # Hidden layer
            with tf.name_scope("hidden_layer"):
                W = tf.Variable(tf.truncated_normal([num_filters_total, 300], stddev=0.1))
                b = tf.Variable(tf.constant(0.1, shape=[300]))
                self.h = tf.nn.tanh(tf.matmul(self.h_pool_flat, W) + b)

            # Add embedding layer
            with tf.name_scope("add_embedding"):
               W = tf.Variable(tf.truncated_normal([300, 300], stddev=0.1), name="W")
               self.y = tf.nn.tanh(tf.matmul(self.h, W), name="tanh")
            with tf.name_scope("dropout"):
                self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

            # Output
            with tf.name_scope("output"):
                W = tf.Variable(tf.truncated_normal([300,2]))
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                self.scores = tf.nn.xw_plus_b(self.y, W, b, name="scores")
                self.predictions = tf.argmax(self.scores, 1, name="predictions")


            # Classify loss
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
                self.classify_loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

            # Accuracy
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            #wirter = tf.summary.FileWriter('submit/', tf.get_default_graph())






