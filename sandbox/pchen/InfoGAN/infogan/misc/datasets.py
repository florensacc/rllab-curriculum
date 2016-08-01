import numpy as np
from tensorflow.examples.tutorials import mnist
import os
import numpy as np


class Dataset(object):
    def __init__(self, images, labels=None):
        self._images = images.reshape(images.shape[0], -1)
        self._labels = labels
        self._epochs_completed = -1
        self._num_examples = images.shape[0]
        # shuffle on first run
        self._index_in_epoch = self._num_examples

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            if self._labels is not None:
                self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        if self._labels is None:
            return self._images[start:end], None
        else:
            return self._images[start:end], self._labels[start:end]


class FaceDataset(object):
    def __init__(self):
        self._data = None
        self._train = None
        self.image_dim = 32 * 32
        self.image_shape = (32, 32, 1)

    @property
    def train(self):
        if self._train is None:
            self._train = Dataset(self.transform(self.data[:, :, :, np.newaxis]))
        return self._train

    @property
    def data(self):
        if self._data is None:
            self._data = np.load(
                "/media/NAS_SHARED/rocky/data/TRANSFORMATION_DATASET/th_AZ_VARIED/FT_training/32_shuffled.npy")
        return self._data

    def transform(self, data):
        # rescale the data to be distributed between 0 and 1
        return data / 255.0

    def inverse_transform(self, data):
        return data * 255.0


class ChairDataset(object):
    def __init__(self):
        self.image_dim = 64 * 64 * 1
        self.image_shape = (64, 64, 1)
        self._data = None
        self._train = None

    @property
    def data(self):
        if self._data is None:
            self._data = np.load("/media/NAS_SHARED/rocky/data/resized_chairs/resized_64_bilinear/color.npy")
            # make data grayscale
            self._data = np.mean(self._data, axis=3)[:, :, :, np.newaxis]
        return self._data

    @property
    def train(self):
        if self._train is None:
            self._train = Dataset(self.transform(self._data))
        return self._train

    def transform(self, data):
        # rescale the data to be distributed between 0 and 1
        return data / 255.0

    def inverse_transform(self, data):
        return data * 255.0


class LessCatChairDataset(object):
    def __init__(self, folder="10cat_resized_64_bilinear"):
        data = np.load("/media/NAS_SHARED/rocky/data/%s/data.npz" % folder)['arr_0']
        # make data grayscale
        data = np.mean(data, axis=3)[:, :, :, np.newaxis]
        # center the data
        self.train = Dataset(self.transform(data))
        self.image_dim = 64 * 64 * 1
        self.image_shape = (64, 64, 1)

    def transform(self, data):
        # rescale the data to be distributed between 0 and 1
        return data / 255.0

    def inverse_transform(self, data):
        return data * 255.0


class MnistDataset(object):
    def __init__(self):
        data_directory = "MNIST"
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
        dataset = mnist.input_data.read_data_sets(data_directory)
        self.train = dataset.train
        # make sure that each type of digits have exactly 10 samples
        sup_images = []
        sup_labels = []
        rnd_state = np.random.get_state()
        np.random.seed(0)
        for cat in range(10):
            ids = np.where(self.train.labels == cat)[0]
            np.random.shuffle(ids)
            sup_images.extend(self.train.images[ids[:10]])
            sup_labels.extend(self.train.labels[ids[:10]])
        np.random.set_state(rnd_state)
        self.supervised_train = Dataset(
            np.asarray(sup_images),
            np.asarray(sup_labels),
        )
        self.test = dataset.test
        self.validation = dataset.validation
        self._image_dim = 28 * 28
        self._image_shape = (28, 28, 1)

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data

    @property
    def image_dim(self):
        return self._image_dim
    @property
    def image_shape(self):
        return self._image_shape

'''
Binarized MNIST (by Hugo Larochelle)
'''
def mnist_binarized(validset=False, flattened=True):
    G.ndict.shuffle(train)
    G.ndict.shuffle(test)
    G.ndict.shuffle(valid)
    if not flattened:
        for data in [train,valid,test]:
            data['x'] = data['x'].reshape((-1,1,28,28))
    if not validset:
        print "Full training set"
        train['x'] = np.concatenate((train['x'], valid['x']))
        return train, test
    return train, valid, test

class BinarizedMnistDataset(object):
    def __init__(self):
        path = "./data" +'/mnist_binarized/'
        import h5py
        train = h5py.File(path+"binarized_mnist-train.h5")['data'][:]#.astype('uint8')*255
        valid = h5py.File(path+"binarized_mnist-valid.h5")['data'][:]#.astype('uint8')*255
        test = h5py.File(path+"binarized_mnist-test.h5")['data'][:]#.astype('uint8')*255

        self.train = Dataset(train)
        self.test = Dataset(valid)
        self.validation = Dataset(test)
        self.image_dim = 28 * 28
        self.image_shape = (28, 28, 1)

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data

