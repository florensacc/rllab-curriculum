from rllab.misc import autoargs


class Algorithm(object):

    @classmethod
    @autoargs.add_args
    def add_args(cls, parser):
        pass

    @classmethod
    @autoargs.new_from_args
    def new_from_args(cls, args):
        pass


class RLAlgorithm(Algorithm):

    def train(self, mdp, policy, vf):
        raise NotImplementedError
