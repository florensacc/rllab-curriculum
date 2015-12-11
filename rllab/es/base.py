from rllab.misc import autoargs


class ExplorationStrategy(object):

    def get_action(self, t, observation, **kwargs):
        raise NotImplementedError

    def episode_reset(self):
        pass

    @classmethod
    @autoargs.add_args
    def add_args(cls, parser):
        pass

    @classmethod
    @autoargs.new_from_args
    def new_from_args(cls, args):
        pass
