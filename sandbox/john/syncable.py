class Syncable(object):
    """
    Algorithm that can be synchronized between multiple instances
    """
    def get_params(self):
        raise NotImplementedError
    def set_params(self, d):
        raise NotImplementedError
    def train_for(self, n_itr):
        raise NotImplementedError