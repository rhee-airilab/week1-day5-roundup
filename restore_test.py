#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
import numpy as np
import tensorflow as tf

data_dir = './mnist'
data_dir

images         = np.fromfile(data_dir + 
                     '/train-images-idx3-ubyte',dtype=np.uint8)
images         = images[16:].reshape([-1,28,28]).astype(np.float32)
images         = images / 127.0 - 1.0

labels         = np.fromfile(data_dir + 
                     '/train-labels-idx1-ubyte',dtype=np.uint8)
labels         = labels[8:].astype(np.int64)


def infer(image, label):

    tf.train.import_meta_graph('save/graph')

    config = tf.ConfigProto(allow_soft_placement=True, gpu_options={'allow_growth': True})
    with tf.Session(config=config) as session:

        session.run(tf.global_variables_initializer())

        ##################################################
        # save/example 에 저장한 파일로 부터 복원
        ##################################################

        saver = tf.train.Saver()

        # saver.restore(session, 'save/example-9361')

        checkpoint = tf.train.latest_checkpoint('save')
        saver.restore(session, checkpoint)

        model_input = 'data/input:0'
        model_pred  = 'prediction/pred:0'

        pred = session.run(
                    model_pred,
                    {model_input:[image.reshape([28*28])]})

        print('infer: label={}, pred={}'.format(label,pred[0]))

if __name__ == '__main__':
    infer(images[1234],labels[1234])
