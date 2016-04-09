from .dataset import Dataset
import numpy as np


class SineDataset(Dataset):

    def __init__(self, train_size, test_size):
        train_x = np.sort(np.random.uniform(-1, 1, size=train_size))
        test_x = np.sort(np.random.uniform(-2, 2, size=test_size))

        x = np.concatenate([train_x, test_x])
        ep = np.random.normal(0, 0.02, size=x.shape)

        y = x + 0.3 * np.sin(2. * np.pi * (x + ep)) + \
            0.3 * np.sin(4. * np.pi * (x + ep)) + ep

        train_y = y[:len(train_x)]
        test_y = y[len(train_x):]

        self._train_x = train_x.reshape((-1, 1))
        self._train_y = train_y.reshape((-1, 1))
        self._test_x = test_x.reshape((-1, 1))
        self._test_y = test_y.reshape((-1, 1))

    @property
    def train_x(self):
        return self._train_x

    @property
    def train_y(self):
        return self._train_y

    @property
    def test_x(self):
        return self._test_x

    @property
    def test_y(self):
        return self._test_y
