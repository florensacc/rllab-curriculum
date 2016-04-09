class Dataset(object):

    @property
    def train_x(self):
        raise NotImplementedError

    @property
    def train_y(self):
        raise NotImplementedError

    @property
    def test_x(self):
        raise NotImplementedError

    @property
    def test_y(self):
        raise NotImplementedError

    @property
    def input_shape(self):
        return self.train_x.shape[1:]

    @property
    def output_shape(self):
        return self.train_y.shape[1:]
