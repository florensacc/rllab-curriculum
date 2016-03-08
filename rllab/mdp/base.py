from rllab.misc import autoargs
import numpy as np
from rllab.misc.ext import flatten_shape_dim


class MDP(object):

    timestep = 0.05

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    @property
    def action_dim(self):
        raise NotImplementedError

    @property
    def observation_shape(self):
        raise NotImplementedError

    @property
    def action_dtype(self):
        raise NotImplementedError

    @property
    def action_bounds(self):
        raise NotImplementedError

    @property
    def observation_dim(self):
        return flatten_shape_dim(self.observation_shape)

    @property
    def observation_dtype(self):
        raise NotImplementedError

    def start_viewer(self):
        pass

    def stop_viewer(self):
        pass

    def plot(self, states=None, actions=None, pause=False):
        raise NotImplementedError

    @classmethod
    @autoargs.add_args
    def add_args(cls, parser):
        pass

    @classmethod
    @autoargs.new_from_args
    def new_from_args(cls, args):
        pass

    def log_extra(self, paths):
        """
        Log extra information per iteration based on the collected paths
        """
        pass

    def print_stats(self):
        print "MDP:\t%s" % self.__class__.__name__
        print "Observation dim:\t%d" % np.prod(self.observation_shape)
        print "Action dim:\t%d" % self.action_dim
