import numpy as np


def load_dataset_MNIST():
    """MNIST dataset loader"""

    import gzip

    def load_mnist_images(filename):
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
            data = data.reshape(-1, 1)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    PATH = '/media/ssd/MNIST/'
    import os
    if not os.path.isdir(PATH):
        PATH = '/Users/rein/openai/datasets/mnist/'

    X_train = load_mnist_images(PATH + 'train-images-idx3-ubyte.gz')
    X_test = load_mnist_images(PATH + 't10k-images-idx3-ubyte.gz')

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train[:256], X_test[:256]


def load_dataset_Atari():
    import pickle

    path = 'sandbox/rein/datasets'
    file_handler = open(path + '/dataset.pkl', 'r')
    _dataset = pickle.load(file_handler)
    return _dataset['x'].transpose(0, 3, 1, 2), _dataset['y'].transpose(0, 3, 1, 2)
