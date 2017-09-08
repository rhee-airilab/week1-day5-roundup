#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import sklearn

from mnist_data import load_mnist

data_dir = './mnist'
train_size = 60000
images, labels = load_mnist(data_dir)
images = images / 127.0 - 1.0
images = np.reshape(images,[-1,28*28])

input_size = 28 * 28
output_size = 10

class MnistClassifier(sklearn.base.BaseEstimator):
    """
    """

    def __init__(   self, \
                    batch_size = 128,
                    hidden_units = 300,
                    max_epoch = 100,
                    activation = 'relu',
                    input_dropout = 0.2,
                    hidden_dropout = 0.5,
                    learning_rate = 0.05,
                    grad_clip = 5.0   ):

        self.batch_size = batch_size
        self.hidden_units = hidden_units
        self.max_epoch = max_epoch
        self.activation = activation
        self.input_dropout = input_dropout
        self.hidden_dropout = hidden_dropout
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip

        self.trained = False


    def fit(self,X,y,**kwargs):

        print(repr(('fit',vars(self))))

        batch_size = self.batch_size
        hidden1_size = self.hidden_units
        hidden2_size = self.hidden_units
        hidden3_size = self.hidden_units
        hidden4_size = self.hidden_units
        max_epoch = self.max_epoch
        activation = self.activation
        input_dropout = self.input_dropout
        hidden_dropout = self.hidden_dropout
        learning_rate = self.learning_rate
        grad_clip = self.grad_clip

        f_gated = False
        if activation.startswith('relu'):
            f_activation = tf.nn.relu

        if activation.startswith('sigm'):
            f_activation = tf.nn.sigmoid

        if activation.startswith('tanh'):
            f_activation = tf.nn.tanh

        if activation.startswith('gated'):
            f_activation = tf.nn.relu
            f_gated = True

        k_init = lambda shape,**kwargs: tf.truncated_normal(shape, stddev=0.02)
        b_init = lambda shape,**kwargs: tf.zeros(shape)
        g_init = lambda shape,**kwargs: tf.truncated_normal(shape, stddev=0.5)

        tf.reset_default_graph()

        with tf.device('/gpu:0'):
            input_ = tf.placeholder(shape=[None, input_size], dtype=tf.float32, name="input")
            label_ = tf.placeholder(shape=[None], dtype=tf.int64, name="label")
            is_training_ = tf.placeholder(dtype=tf.bool, name="is_training")

            with tf.name_scope("fc1"):
                hidden1 = tf.layers.dense(input_, hidden1_size, kernel_initializer=k_init, bias_initializer=b_init, activation=f_activation)
                hidden1 = tf.layers.dropout(hidden1, rate=input_dropout, training=is_training_, name="dropout1")
                if f_gated:
                    hidden1_g = tf.layers.dense(input_, hidden1_size, kernel_initializer=k_init, bias_initializer=g_init, activation=tf.nn.sigmoid)
                    hidden1 = hidden1 * hidden1_g

            with tf.name_scope("fc2"):
                hidden2 = tf.layers.dense(hidden1, hidden2_size, kernel_initializer=k_init, bias_initializer=b_init, activation=f_activation)
                hidden2 = tf.layers.dropout(hidden2, rate=hidden_dropout, training=is_training_, name="dropout2")
                if f_gated:
                    hidden2_g = tf.layers.dense(hidden1, hidden2_size, kernel_initializer=k_init, bias_initializer=g_init, activation=tf.nn.sigmoid)
                    hidden2 = hidden2 * hidden2_g

            with tf.name_scope("fc3"):
                hidden3 = tf.layers.dense(hidden2, hidden3_size, kernel_initializer=k_init, bias_initializer=b_init, activation=f_activation)
                hidden3 = tf.layers.dropout(hidden3, rate=hidden_dropout, training=is_training_, name="dropout3")
                if f_gated:
                    hidden3_g = tf.layers.dense(hidden2, hidden3_size, kernel_initializer=k_init, bias_initializer=g_init, activation=tf.nn.sigmoid)
                    hidden3 = hidden3 * hidden3_g

            with tf.name_scope("fc4"):
                hidden4 = tf.layers.dense(hidden3, hidden4_size, kernel_initializer=k_init, bias_initializer=b_init, activation=f_activation)
                hidden4 = tf.layers.dropout(hidden4, rate=hidden_dropout, training=is_training_, name="dropout4")
                if f_gated:
                    hidden4_g = tf.layers.dense(hidden3, hidden4_size, kernel_initializer=k_init, bias_initializer=g_init, activation=tf.nn.sigmoid)
                    hidden4 = hidden4 * hidden4_g

            with tf.name_scope("fc5"):
                output  = tf.layers.dense(hidden4, output_size, kernel_initializer=k_init, bias_initializer=b_init)
                if f_gated:
                    output_g = tf.layers.dense(hidden4, output_size, kernel_initializer=k_init, bias_initializer=g_init, activation=tf.nn.sigmoid)
                    output = output * output_g
                pred = tf.nn.softmax(output)

            loss = tf.losses.sparse_softmax_cross_entropy(label_, output)
            trainer = tf.train.GradientDescentOptimizer(learning_rate)

            if f_gated:
                tvars           = tf.trainable_variables()
                grads, _        = tf.clip_by_global_norm(tf.gradients(loss, tvars, colocate_gradients_with_ops=True), grad_clip)
                optimize        = trainer.apply_gradients(zip(grads, tvars), name='step')
            else:
                optimize = trainer.minimize(loss)

            pred_ = tf.argmax(pred, axis=1)
            correct = tf.equal(label_,pred_)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

            check_op_ = [tf.check_numerics(t, "check numerics: grads") for t in tf.gradients(loss, tf.trainable_variables())]


        config = tf.ConfigProto(allow_soft_placement=True, gpu_options={'allow_growth': True})
        session = tf.Session(config=config)

        init = tf.global_variables_initializer()
        session.run(init)

        for epoch in range(max_epoch):
            #train
            losses, accs = [], []
            for i in range(0, train_size, batch_size):
                img = images[i:i+batch_size]
                lbl = labels[i:i+batch_size]
                _, loss_v, acc_v, _ = session.run([optimize, loss, accuracy, check_op_], feed_dict= {input_: img, label_: lbl, is_training_: True})
                losses.append(loss_v)
                accs.append(acc_v)
            print('Epoch {} loss: {} acc: {}'.format(epoch, np.mean(losses), np.mean(accs)))

        self.accuracy = accuracy
        self.input_ = input_
        self.label_ = label_
        self.is_training_ = is_training_

        self.session = session
        self.trained = True

        return self


    def score(self,X,y):

        print(repr(('score',vars(self))))

        session = self.session

        t_images = X
        t_labels = y
        test_size = len(t_images)

        # test
        t_accs = []
        for i in range(0, test_size, self.batch_size):
            img = t_images[i:i + self.batch_size]
            lbl = t_labels[i:i + self.batch_size]
            acc = session.run(self.accuracy, feed_dict={self.input_: img, self.label_: lbl, self.is_training_: False})
            t_accs.append(acc)
        total_acc = np.mean(t_accs)

        print('  Test Score: {}%'.format(100.0 * total_acc))

        return total_acc


if __name__ == '__main__':
    estimator = MnistClassifier()
    estimator.fit(None,None)
    print(repr(('score',estimator.score(None,None))))

