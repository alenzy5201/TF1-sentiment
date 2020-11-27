#! /usr/bin/env python
from tensorflow import keras
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helper
from text_cnn_4 import TextCNN
from tensorflow.contrib import learn

TRAIN_EMBEDDING_EPOCH = 10000
TRAIN_CLASSIFY_EPOCH = 10000

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("w2v_path", "./w2v_model/GoogleNews-vectors-negative300.bin", "w2v file")
tf.flags.DEFINE_string("file_dir", ".\Large_Scale_Sentiment_Classification_Data\Weakly_labeled_data_1.1M", "embedding_train dataSet")
tf.flags.DEFINE_string("classify_file_dir", "./Large_Scale_Sentiment_Classification_Data/Labeled_data_11754/11754.csv", "train dataSet")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "1,2,3", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 200, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.0, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
def preprocess():
    # Load sentences
    print("Loading data...")
    file_names = os.listdir(FLAGS.file_dir)
    neg_sentences, neg_labels, pos_sentences, pos_labels \
        =data_helper.load_data_and_labels(file_dir=FLAGS.file_dir, file_names=file_names)
    classify_sentences, y = data_helper.load_classify_data(FLAGS.classify_file_dir)
    print("Loading data finish")


    max_document_length = data_helper.max_sentence_length([neg_sentences, pos_sentences, classify_sentences])
    print('句子的最大长度为：{}'.format(max_document_length))

    #Build vocab
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    combine_x = np.array(list(vocab_processor.fit_transform(pos_sentences + neg_sentences + classify_sentences)))
    x_pos = combine_x[:len(pos_sentences)]
    x_neg = combine_x[len(pos_sentences): len(pos_sentences) + len(neg_sentences)]
    x = combine_x[len(pos_sentences) + len(neg_sentences):]
    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))

    return x, y, vocab_processor,x_pos,x_neg

def train(x, y, vocab_processor,lr, dropout, l2_lambda,x_pos,x_neg):
    with tf.Graph().as_default():
        test = 10
        if test:
            W = tf.random_normal([len(list(vocab_processor.vocabulary_._mapping)), FLAGS.embedding_dim], stddev=17,
                                 mean=23)
            # np.random.normal(loc=0,scale=2,size=[len(list(vocab_processor.vocabulary_._mapping)), FLAGS.embedding_dim])
        else:
            W = data_helper.get_W(vocab=list(vocab_processor.vocabulary_._mapping),
                                  fname=FLAGS.w2v_path,
                                  voacb_ids_map=vocab_processor.vocabulary_._mapping
                                  )
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        sess_config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True,allow_soft_placement=True)
        #with tf.Session(config=sess_config) as sess:
        sess = tf.Session(config=sess_config)
        classify_global_step = tf.Variable(0, name="global_step", trainable=False)

        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x.shape[1],
                num_classes=2,
                embedding_table=W,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            embedding_optimizer = tf.train.AdamOptimizer(lr)
            grads_and_vars =embedding_optimizer.compute_gradients(cnn.embedding_loss)
            embedding_train_op = embedding_optimizer.apply_gradients(grads_and_vars, global_step=classify_global_step)

            classify_optimizer = tf.train.AdamOptimizer(lr)#-6可以
            grads_and_vars =  classify_optimizer.compute_gradients(cnn.classify_loss)
            train_op =  classify_optimizer.apply_gradients(grads_and_vars, global_step= classify_global_step)

            # Keep track of gradient values and sparsity
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    value_hist_summary = tf.summary.histogram("{}/value/hist".format(v.name), v)
                    sparsity_value_summary = tf.summary.scalar("{}/value/sparsity".format(v.name), tf.nn.zero_fraction(v))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
                    grad_summaries.append(value_hist_summary)
                    grad_summaries.append(sparsity_value_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Summaries for loss and accuracy
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
            classify_loss_summary_classify = tf.summary.scalar("classify_loss", cnn.classify_loss)
            embedding_loss_summary = tf.summary.scalar("embedding_loss",cnn.embedding_loss)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.getcwd(), "runs_e", 'lr={}\drop={}\l2_lambda={}'.format(lr,dropout,l2_lambda)))
            print("Writing to {}\n".format(out_dir))

            # Train Summaries
            embedding_summary_op = tf.summary.merge([embedding_loss_summary, grad_summaries_merged])
            embedding_summary_dir = os.path.join(out_dir, "summaries", "embedding")
            embedding_summary_writer = tf.summary.FileWriter( embedding_summary_dir, sess.graph)

            train_summary_op = tf.summary.merge([acc_summary,classify_loss_summary_classify, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
            dev_summary_op = tf.summary.merge([classify_loss_summary_classify, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def  train_embedding(x_batch):
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: [[1, 0],
                                  [1, 0],
                                  [1, 0]],
                }
                _, step, loss, accuracy, summaries = sess.run(
                    [embedding_train_op, classify_global_step, cnn.embedding_loss, cnn.accuracy,embedding_summary_op],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("Embedding training {}: step {}, loss {:g} accuracy {}".format(time_str, step, loss, accuracy))
                embedding_summary_writer.add_summary(summaries, step)

            def train_classify(x, y):
                feed_dict = {
                    cnn.input_x: x,
                    cnn.input_y: y,
                }
                _, step, loss, accuracy, summaries = sess.run(
                    [train_op, classify_global_step, cnn.classify_loss, cnn.accuracy,train_summary_op],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("Classify training {}: step {}, loss {:g} accuracy {}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def test_classify(x_test, y_test):
                feed_dict = {
                    cnn.input_x: x_test,
                    cnn.input_y: y_test,
                    cnn.dropout_keep_prob:1
                }
                step, loss, accuracy, summaries = sess.run(
                    [classify_global_step, cnn.classify_loss, cnn.accuracy, dev_summary_op],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("Test:  {}: step {}, loss {:g} accuracy {}".format(time_str, step, loss, accuracy))
                dev_summary_writer.add_summary(summaries, step)

            #Tune
            VAL_NUM = 64
            shuffle_index = np.random.permutation(len(x))
            x = x[shuffle_index]
            y = y[shuffle_index]
            x_train = x[VAL_NUM:]
            y_train = y[VAL_NUM:]
            x_test = x[:VAL_NUM]
            y_test = y[:VAL_NUM]
            test2 = 0
            if test2:
                (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=8000)
                y1tmp = []
                y2tmp = []
                for i in y_train:
                    if i==1:
                        y1tmp.append([1,0])
                    elif i ==0:
                        y1tmp.append([0,1])
                for i in y_test:
                    if i==1:
                        y2tmp.append([1,0])
                    elif i ==0:
                        y2tmp.append([0,1])
                y_train = np.array(y1tmp)
                y_test  = np.array(y2tmp)
                x_train = keras.preprocessing.sequence.pad_sequences(x_train,
                                                                     maxlen=130, padding='post')
                x_test = keras.preprocessing.sequence.pad_sequences(x_test,
                                                                    maxlen=130, padding='post')

            # Train embedding layer
            train_embedding_epoch = TRAIN_EMBEDDING_EPOCH
            for i in range(train_embedding_epoch):
                x_batch = data_helper.embedding_batch(x_pos, x_neg)
                train_embedding(x_batch=x_batch)

            for i in range(TRAIN_CLASSIFY_EPOCH):
                x_train_batch, y_train_batch = data_helper.classify_batch(x_train,y_train,batch_size=32)
                train_classify(x_train_batch, y_train_batch)
                #test every 100 times
                if i % 10 == 0:
                    x_test_batch, y_test_batch = data_helper.classify_batch(x_test,y_test,batch_size=64)
                    test_classify(x_test_batch, y_test_batch)

def main(argv=None):
    x, y, vocab_processor,x_pos,x_neg= preprocess()
    lr = 1e-2
    dropout = 0.8
    l2_lambda = 100
    train(x, y, vocab_processor, lr, dropout, l2_lambda, x_pos, x_neg)
    # for lr in [1/pow(10,x) for x in range(5,11)]:
    #     for dropout in [x/10 for x in range(10,6,-2)]:
    #         for l2_lambda in [50,5,0]:
    #             train(x, y, vocab_processor, lr, dropout, l2_lambda,W)\
    # lr = 1e-3
    # for lr in [1e-3]:
    #     for dropout in [x/10 for x in range(10, 6, -1)]:
    #         for l2_lambda in range(100,0,-20):
    #             train(x, y, vocab_processor, lr, dropout, l2_lambda,x_pos,x_neg)


if __name__ == '__main__':
    tf.app.run()
