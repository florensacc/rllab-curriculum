class ValueFunction(object):

    def get_param_values(self):
        raise NotImplementedError

    def set_param_values(self, val):
        raise NotImplementedError

    def fit(self, paths):
        raise NotImplementedError

    def predict(self, path):
        raise NotImplementedError

    @classmethod
    def add_args(cls, parser):
        pass

    @classmethod
    def new_from_args(cls, args, mdp):
        return cls()
