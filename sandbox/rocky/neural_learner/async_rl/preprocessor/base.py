class Preprocessor(object):
    def __init__(self):
        pass

    @property
    def input_dim(self):
        return None

    @property
    def output_dim(self):
        return None

    def process(self,items):
        raise NotImplementedError
