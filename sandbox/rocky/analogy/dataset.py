from __future__ import print_function
from __future__ import absolute_import
import numpy as np


class BatchDataset(object):
    def __init__(self, inputs, batch_size, extra_inputs=None, shuffler=None, input_keys=None):
        self.shuffler = shuffler
        self.inputs = list(inputs)
        if extra_inputs is None:
            extra_inputs = []
        self.extra_inputs = extra_inputs
        self.batch_size = batch_size
        self.input_keys = input_keys
        if batch_size is not None:
            self.ids = np.arange(self.inputs[0].shape[0])
            self.update()

    @property
    def input_dict(self):
        assert self.input_keys is not None
        return dict(zip(self.input_keys, self.inputs + self.extra_inputs))


    @property
    def number_batches(self):
        if self.batch_size is None:
            return 1
        return int(np.ceil(self.inputs[0].shape[0] * 1.0 / self.batch_size))

    def iterate(self, update=True, return_dict=False):
        if return_dict:
            assert self.input_keys is not None
        if self.batch_size is None:
            yield list(self.inputs) + list(self.extra_inputs)
        else:
            for itr in range(self.number_batches):
                batch_start = itr * self.batch_size
                batch_end = (itr + 1) * self.batch_size
                batch_ids = self.ids[batch_start:batch_end]
                batch = [d[batch_ids] for d in self.inputs]
                if return_dict:
                    yield dict(zip(self.input_keys, list(batch) + list(self.extra_inputs)))
                else:
                    yield list(batch) + list(self.extra_inputs)
            if update:
                self.update()

    def update(self):
        np.random.shuffle(self.ids)
        if self.shuffler is not None:
            self.shuffler.shuffle(*self.inputs)


class SupervisedDataset(object):
    def __init__(self, inputs, train_batch_size, train_ratio, extra_inputs=None, test_batch_size=1,
                 shuffler=None, input_keys=None):
        if extra_inputs is None:
            extra_inputs = []
        n_total = len(inputs[0])
        n_train = int(np.ceil(n_total * train_ratio))
        assert n_train < n_total, "Insufficient data! %d * %.2f < 1" % (n_total, 1 - train_ratio)
        train_inputs = [x[:n_train] for x in inputs]
        test_inputs = [x[n_train:] for x in inputs]

        self.input_keys = input_keys

        self.train = BatchDataset(inputs=train_inputs, batch_size=train_batch_size, extra_inputs=extra_inputs,
                                  shuffler=shuffler, input_keys=input_keys)
        self.test = BatchDataset(inputs=test_inputs, batch_size=test_batch_size, extra_inputs=extra_inputs,
                                 shuffler=shuffler, input_keys=input_keys)
