import os
import numpy as np

def load_mnist(path):
    fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
    images = np.fromfile(file=fd, dtype=np.uint8)
    images = images[16:].reshape([60000, 28, 28]).astype(np.float)

    fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
    labels = np.fromfile(file=fd, dtype=np.uint8)
    labels = labels[8:].reshape([60000]).astype(np.int)

    return images, labels

def load_mnist_t10k(path):
    fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
    images = np.fromfile(file=fd, dtype=np.uint8)
    images = images[16:].reshape([10000, 28, 28]).astype(np.float)

    fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
    labels = np.fromfile(file=fd, dtype=np.uint8)
    labels = labels[8:].reshape([10000]).astype(np.int)

    return images, labels

"""
# TEST DRIVE
images, labels = load_mnist('./mnist')
save_images('output.png', images[0:64], [8, 8])
print(labels[0:16])
"""
