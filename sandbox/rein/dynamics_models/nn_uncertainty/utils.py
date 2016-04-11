import numpy as np

# This function takes an array of numbers and smoothes them out.
# Smoothing is useful for making plots a little easier to read.


def sliding_mean(data_array, window=5):
    data_array = np.array(data_array)
    new_list = []
    for i in range(len(data_array)):
        indices = range(max(i - window + 1, 0),
                        min(i + window + 1, len(data_array)))
        avg = 0
        for j in indices:
            avg += data_array[j]
        avg /= float(len(indices))
        new_list.append(avg)

    return np.array(new_list)


def load_dataset_MNIST():
    """MNIST dataset loader"""

    import gzip

    def load_mnist_images(filename):
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 28 * 28)
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
        PATH = '/home/rein/PhD/MNIST/'

    X_train = load_mnist_images(PATH + 'train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels(PATH + 'train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images(PATH + 't10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels(PATH + 't10k-labels-idx1-ubyte.gz')

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_test, y_test


def load_dataset_1Dregression():
    """Synthetic 1D regression data loader"""

    def generate_synthetic_data(dat_size=1.e4):
        rng = np.random.RandomState(1234)
        x = rng.uniform(0.05, 0.95, dat_size).reshape((dat_size, 1))
        v = rng.normal(0, 0.02, size=x.shape)
        # TODO: Write y.
        y = x + 0.3 * np.sin(2. * np.pi * (x + v)) + 0.3 * \
            np.sin(4. * np.pi * (x + v)) + v
#         y += np.random.randint(low=0, high=2, size=(len(y), 1))
#         rand = np.zeros((len(x), 1))
#         for i in xrange(rand.shape[0]):
#             rand[i] = random.choice([-1,1])
#         x += rand * 2
        # 90% for training, 10% testing
        train_x = x[:len(x) * 0.9]
        train_y = y[:len(y) * 0.9]
        test_x = x[len(x) * 0.9:]
        test_y = y[len(y) * 0.9:]

        x2 = rng.uniform(1., 11., dat_size).reshape((dat_size, 1))
        y2 = x2 * 0. + 0.5
        test_x = np.vstack([test_x, x2])
        test_y = np.vstack([test_y, y2])
        x2 = rng.uniform(-9., 1., dat_size).reshape((dat_size, 1))
        y2 = x2 * 0. + 0.5
        test_x = np.vstack([x2, test_x])
        test_y = np.vstack([y2, test_y])
        
#         print(train_x.shape)
        train_x = np.hstack([train_x, train_x**2, train_x**3, train_x**4])
        test_x = np.hstack([test_x, test_x**2, test_x**3, test_x**4])
#         print(new_train_x.shape)
#         print(new_train_x[:,0][:,None].shape)
#         print(train_x == new_train_x[:,0][:,None])
        return (train_x, train_y), (test_x, test_y)

    return generate_synthetic_data(1e3)


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]