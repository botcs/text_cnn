#! /usr/bin/env python3
# coding: utf-8

import tensorflow as tf
import numpy as np
import os

# Model Hyperparameters
flags = tf.app.flags
flags.DEFINE_integer('batch_size', 16, 'Number of samples per training cycle [16]')
flags.DEFINE_integer('embedding_size', 10, 'Dimension of vector space each word is projected into [10]')
flags.DEFINE_integer('num_filters', 7, 'Number of filters to use for each feature extractor size [16]')
flags.DEFINE_string('filter_size', '3, 4, 5', 'Filter size will determine how many words each filter extractor will examine at once. Use comma separated integers ["3, 4, 5"]')
flags.DEFINE_integer('max_pool', 3, 'K strongest activations are selected of each feature extractor [3]')
flags.DEFINE_string('hidden_size', '50, 30', 'Width of classifier fully connected layers. Use comma separated integers ["50, 30"]')
flags.DEFINE_float('keep_prob', 0.5, 'Probability of keeping an activation value after the DROPOUT layer, during training [0.5]')
flags.DEFINE_string('log_dir', '/tmp/log', 'Logs will be saved to this directory')

FLAGS = flags.FLAGS
FLAGS._parse_flags()

import data

class model():
    def get_reader(self, batch_size):
        num_classes = data.num_classes
        def parse_example(filename_queue):
            # Define how to parse the example
            
            reader = tf.TFRecordReader()
            _, example = reader.read(filename_queue)
            
            context_features = {
                'length': tf.FixedLenFeature([1], dtype=tf.int64),
                'labels': tf.FixedLenFeature([num_classes], dtype=tf.int64)
            }
            sequence_features = {
                "tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64)
            }
            context_parsed, sequence_parsed = tf.parse_single_sequence_example(
                serialized=example,
                context_features=context_features,
                sequence_features=sequence_features
            )
            return context_parsed, sequence_parsed

        filename_queue = tf.train.string_input_producer([
            os.path.join(FLAGS.data_dir, 'cnn.TFRecord')])
        context_parsed, sequence_parsed = parse_example(filename_queue)
        

        x = tf.train.batch(
            name='context_reader',
            tensors=sequence_parsed,
            batch_size=batch_size,
            dynamic_pad=True)['tokens']


        y = tf.train.batch(
            name='genre_reader',
            tensors=context_parsed,
            batch_size=batch_size)['labels']

        x = tf.cast(x, tf.int32)
        y = tf.cast(y, tf.float32)
        
        return x, y
    
    def get_embedding(self, x, batch_size, embedding_size):
        
        vocabulary_size = data.vocabulary_size
        
        with tf.device('/cpu'):
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, x)
            # BatchSize, SeqLen, EmbeddSize, Channels
            embed = tf.reshape(embed, [batch_size, -1, embedding_size, 1])

        return embed
    
    def get_feature(self, embed, kernel_heights, num_filters, k_max):
        # Convolution layers acting as feature extractors
        # kernel is wide as each word embedding representation
        # kernel height defines how much word will it take into account
        embedding_size = embed.get_shape().as_list()[2]
        max_pooled = []
        for height in kernel_heights:
            with tf.variable_scope('read_%d_words' % height):
                # Possibly buggy, if not the first dimension is unknown [?, ...]
                conv = tf.contrib.layers.conv2d(embed, num_filters,
                    [height, embedding_size], padding='VALID')
                
                '''
                kernel_shape = [height, embedding_size, 1, num_filters]
                
                W = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.1), 
                    name="kernel")
                    
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), 
                    name="bias")
                '''
                # BatchSize, ConvWordTuples, SingleWordWindow(=1), Features
                conv_tr = tf.transpose(conv, perm=[0, 3, 2, 1])
                pool = tf.nn.top_k(conv_tr, k=k_max, sorted=False,  
                    name=('k_max%d' % k_max)).values
            max_pooled.append(pool)

        feature = tf.concat(3, max_pooled)
        feature = tf.transpose(feature, [0, 3, 1, 2])    
        
        return feature


    def get_classifier(self, feature, hidden_sizes, keep_prob_default):
        # Fix Dimensions
        # 
        # feauture has the following dimensions: 
        #   [batch_size, len(kernel_heights), num_filters, k_max]
        # 
        # Which means each variable length sequence sample is parsed by a 
        # convolution kernel taking in account kernel_height number of 
        # neighbouring embedded word vectors with size embedding_size.
        # 
        # Since this representation is still variable length 
        # seq_len - kernel_height + 1, use k_max pooling, 
        # to obtain fixed dimensional representations for classifying layers.
        #
        # using sigmoid output, for guessing each category

        num_classes = data.num_classes
        keep_prob = tf.placeholder_with_default(
            keep_prob_default, [], name='dropout_keep_prob')

        h = tf.contrib.layers.flatten(feature)
        for width in hidden_sizes:
            with tf.variable_scope('hidden_%d' % width):
                h = tf.contrib.layers.fully_connected(h, width)
                h = tf.nn.dropout(h, keep_prob)

        # logits for cross-entropy
        logits = tf.contrib.layers.fully_connected(
            h, num_classes, activation_fn=None, scope='output_layer')    
        y_pred = tf.sigmoid(logits)
        
        return logits, y_pred, keep_prob
        
    def __init__(self,
        batch_size=FLAGS.batch_size,
        embedding_size=FLAGS.embedding_size,
        num_filters=FLAGS.num_filters,
        filter_size=[int(s) for s in FLAGS.filter_size.split(',')],
        k_max=FLAGS.max_pool,
        hidden=[int(s) for s in FLAGS.hidden_size.split(',')],
        keep_prob_default=FLAGS.keep_prob):
        
        
        print('Graph initialization...')
        # using variable scope, for cleaner TensorBoard representation
        
        with tf.variable_scope('reader'):
            self.x_buffer, self.y = self.get_reader(batch_size)
            self.input = tf.placeholder_with_default(
                self.x_buffer, [None, None])
            print('1. TFRecords reader')
    
        with tf.variable_scope('embedding'):
            self.embed = self.get_embedding(
                self.input, batch_size, embedding_size)
            print('2. Word embedding')
        
        with tf.variable_scope('convolution'):    
            self.feature = self.get_feature(
                self.embed, filter_size, num_filters, k_max)
            print('3. Feature Extractors')

        with tf.variable_scope('classifier'):
            self.logits, self.pred, self.keep_prob = self.get_classifier(
                self.feature, hidden, keep_prob_default)
            print('4. Classifier')
            
             

def main(argv=None):

    graph = model()
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.train.SummaryWriter(
            FLAGS.log_dir, graph=sess.graph)

        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        output = graph.pred.eval()
        assert(output.shape == (FLAGS.batch_size, data.num_classes))
        print('dataflow OK')
        
        coord.request_stop()
        coord.join(threads)
            

    
        
if __name__ == '__main__':
    
    print('Running model.py')
    print('\nParameters:')
    for attr, value in sorted(FLAGS.__flags.items()):
        print('{}={}'.format(attr.upper(), value))
    print('')
    print('checking DATA_DIR')
    if os.path.exists(FLAGS.data_dir):
        print('0. Found DATA_DIR')
    else:    
        data.main()
    tf.app.run()
