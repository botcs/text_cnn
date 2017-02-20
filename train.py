#! /usr/bin/env python3
# coding: utf-8

import tensorflow as tf

# If defined after import of data and model
# Flags are somehow only gets defined when main runs
# maybe multiple imports operating on the same FLAGS object cause this

# Training parameters
flags = tf.app.flags
flags.DEFINE_integer('train_steps', 100, 'Number of batches to train a new model on [100]')
flags.DEFINE_string('optimizer', 'adam', 'Optimizer used to decrease the loss function. Default is adam [adam | sgd]')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate to train the network with [0.001]')
flags.DEFINE_string('checkpoint_file', '/tmp/model.ckpt', 'If file exists, then the loaded model is trained. Else a new model will be trained and saved [./checkpoints/model.ckpt]')
FLAGS = flags.FLAGS


import data
import model
import os

# If this is called, then MIRACOLOUSLY unrolled_lstm will be appear
# WOW!

FLAGS._parse_flags()


class trainer():
        
    def load_maybe(self, model_path, sess):
        if os.path.isfile(model_path+'.index'):
            tf.train.Saver().restore(sess, model_path)
            print('Model loaded from', model_path)
            return True
        else:
            print('Model not found, training new model')
            return False
        
    def save(self, model_path, sess):
        if os.path.isfile(model_path):
            print('Overwriting model file', model_path)
        
        tf.train.Saver().save(sess, model_path)
        print('Model saved to', model_path)
    
    def train_step(self, sess, writer=None):
        # should be called where default_session is defined
        _, step, loss, acc = sess.run(
            [self.opt, self.global_step, self.loss, self.acc])
            
        if writer:
            writer.add_summary(self.summ_op.eval(), step)
        return step, loss, acc    
    
    def eval_step(self, sess, step, writer=None):
        loss, acc = sess.run([self.loss, self.acc])
        
        if writer:
            writer.add_summary(self.summ_op.eval(), step)
            
        return loss, acc
    
    def train(self, 
        checkpoint_file=FLAGS.checkpoint_file,
        train_steps=FLAGS.train_steps):

        sum_acc = 0
        with tf.Session() as sess:
            # Need to run this for initializing reader queue
            coord = tf.train.Coordinator()
            sess.run(tf.global_variables_initializer())
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            
            
            self.writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
            self.load_maybe(checkpoint_file, sess)
            step = start_step = self.global_step.eval()
            
            try:
                while step-start_step < train_steps and not coord.should_stop():
                    step, loss, acc = self.train_step(sess, self.writer)
                    sum_acc += acc
                    
                    if step%20 == 0: print(
                        'global step %06d, loss %1.4f, acc %1.4f'
                        %(step, loss, acc))
            except tf.errors.OutOfRangeError:
                print('Read queue is empty')
            finally:
                # When done, ask the threads to stop.
                print('Done training, overall accuracy:', sum_acc/train_steps)
                self.save(FLAGS.checkpoint_file, sess)
                coord.request_stop()
            
            coord.join(threads)
        
    def evaluate(self,
        checkpoint_file=FLAGS.checkpoint_file,
        eval_steps=FLAGS.train_steps):
        sum_loss = 0
        sum_acc = 0
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            sess.run(tf.global_variables_initializer())
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            if not self.load_maybe(checkpoint_file, sess):
                self.train()
                
            step = start_step = self.global_step.eval()
            print('Evaluating, trained steps', start_step)
            while step-start_step < eval_steps and not coord.should_stop():
                loss, acc = self.eval_step(sess, step, self.writer)
                step += 1
                sum_loss += loss
                sum_acc += acc
                print('\rStep: %d / %d    ' % (step-start_step, eval_steps), 
                    end='', flush=True)
            print('\nDone evaluating, overall loss %1.4f, acc %1.4f'
                %(sum_loss/eval_steps, sum_acc/eval_steps))
                
            coord.request_stop()
            coord.join(threads)
      
    def __init__(self,
        train_steps=FLAGS.train_steps,
        learning_rate=FLAGS.learning_rate):
        
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.model = model.model()
        self.optimizer = tf.train.AdamOptimizer if FLAGS.optimizer == 'adam' \
            else tf.train.GradientDescentOptimizer
        
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.model.logits, targets=self.model.y)
                
        self.loss = tf.reduce_mean(loss)
        self.opt = self.optimizer(FLAGS.learning_rate).minimize(
            loss, global_step=self.global_step)
        
        # Accuracy
        total = FLAGS.batch_size * data.num_classes
        missed = tf.reduce_sum(tf.abs(self.model.y - tf.round(self.model.pred)))
        self.acc = (total - missed) / total
    
        # (Train) Summaries for TensorBoard
        self.loss_summary = tf.summary.scalar('loss', self.loss)
        self.acc_summary = tf.summary.scalar('accuracy', self.acc)
        self.summ_op = tf.summary.merge([self.loss_summary, self.acc_summary])
        
        # Evaluation summaries for TensorBoard
        test_loss_summary = tf.summary.scalar('test_loss', self.loss)
        test_acc_summary = tf.summary.scalar('test_accuracy', self.acc)
        summ_op = tf.summary.merge([test_loss_summary, test_acc_summary])
        

def main(argv=None):
    T = trainer()
    T.train()
    T.evaluate()
    
    

if __name__ == '__main__':
    print('Running train.py')
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

