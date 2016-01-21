from rllab.misc import autoargs


class Options(object):
    pass


class Algorithm(object):

    def __init__(self):
        self.opt = Options()

    @classmethod
    @autoargs.add_args
    def add_args(cls, parser):
        pass

    @classmethod
    @autoargs.new_from_args
    def new_from_args(cls, args):
        pass


class RLAlgorithm(Algorithm):

    def train(self, mdp, **kwargs):
        raise NotImplementedError
