# coding: utf-8
#############################################
# tf.debug example
#############################################

from __future__ import print_function, division, absolute_import

import tensorflow as tf
import numpy as np

DEBUG_   = False

data_dir = './mnist'
data_dir

images = np.fromfile(data_dir + '/train-images-idx3-ubyte',dtype=np.uint8)
images = images[16:].reshape([-1,28,28]).astype(np.float32)
images = images / 127.0 - 1.0

labels = np.fromfile(data_dir + '/train-labels-idx1-ubyte',dtype=np.uint8)
labels = labels[8:].astype(np.int64)

from tensorflow.python import debug as tf_debug

class Model:
    '''
    simple 1-layer fully-connected network for MNIST problem
    '''
    def __init__(self):
        learning_rate = 0.05
        input_size = 28 * 28
        output_size = 10

        input_ = tf.placeholder(shape=[None, input_size], dtype=tf.float32, name="input")
        label  = tf.placeholder(shape=[None], dtype=tf.int64, name="label")

        weights = tf.Variable(tf.zeros([input_size, output_size]))
        biases = tf.Variable(tf.zeros([output_size]))
        output = tf.matmul(input_, weights) + biases

        loss = tf.losses.sparse_softmax_cross_entropy(label,output)
        optimize = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        pred = tf.argmax(output, axis=1, name='pred')
        accuracy = 1.0 - tf.cast(tf.count_nonzero(pred-label),tf.float32) / \
                   tf.cast(tf.size(label),tf.float32)

        self.input = input_
        self.label = label
        self.loss  = loss
        self.optimize = optimize
        self.pred = pred
        self.accuracy = accuracy
        self.weights = weights
        self.biases = biases


def train(max_epochs=20):

    tf.reset_default_graph()

    model = Model()

    batch_size = 128
    batch_count = 60000 // batch_size

    step = 1

    config = tf.ConfigProto(gpu_options={'allow_growth': True})
    with tf.Session(config=config) as session:

        if DEBUG_:
            session = tf_debug.LocalCLIDebugWrapperSession(session)
            session.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)
    
        session.run(tf.global_variables_initializer())
        for ep in range(max_epochs):
            total_loss = 0
            total_acc_v = 0
            for i in range(batch_count):
                img = np.reshape(images[i*batch_size:(i+1)*batch_size], [batch_size, 28 * 28])
                lbl = (labels[i*batch_size:(i+1)*batch_size])
                _, loss_v, acc_v = session.run(
                    [model.optimize, model.loss, model.accuracy],
                    feed_dict= {model.input: img, model.label: lbl})
                step += 1
                total_loss += loss_v
                total_acc_v += acc_v
            print('Epoch %d: training_loss: %.5f training_accuracy: %.3f%%' % (
                ep+1, total_loss / batch_count, total_acc_v / batch_count * 100))

        ##################################################
        # save/graph 에 저장
        ##################################################
            
        tf.train.export_meta_graph(filename='save/graph')

        ##################################################
        # save/example 에 저장
        ##################################################
            
        saver = tf.train.Saver()
        checkpoint = saver.save(session, 'save/example', global_step=step)
        print('Saved: %s'%(checkpoint,))

    return step


def infer(image, label=None):
    
    tf.reset_default_graph()

    model = Model()

    with tf.Session(config=tf.ConfigProto(gpu_options={'allow_growth': True})) as session:

        session.run(tf.global_variables_initializer())

        ##################################################
        # save/example 에 저장한 파일로 부터 복원
        ##################################################

        saver = tf.train.Saver()
        saver.restore(session, 'save/example-9361')
        
        pred = session.run(model.pred,{model.input:[image.reshape([28*28])]})

        print('infer: label={}, pred={}'.format(label,pred[0]))


if __name__ == '__main__':
    import sys
    if '--debug' in sys.argv[1:]:
        DEBUG_ = True
    train(20)
    infer(images[1234],labels[1234])
