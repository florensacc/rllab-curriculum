from rllab.misc import autoargs
from rllab.core.parameterized import Parameterized


class QFunction(Parameterized):

    def __init__(self, mdp):
        self._observation_shape = mdp.observation_shape
        self._observation_dtype = mdp.observation_dtype
        self._action_dim = mdp.action_dim
        self._action_dtype = mdp.action_dtype

    @classmethod
    @autoargs.add_args
    def add_args(cls, parser):
        pass

    @classmethod
    @autoargs.new_from_args
    def new_from_args(cls, args, mdp):
        pass

    @property
    def observation_shape(self):
        return self._observation_shape

    @property
    def observation_dtype(self):
        return self._observation_dtype

    @property
    def action_dim(self):
        return self._action_dim

    @property
    def action_dtype(self):
        return self._action_dtype

    def log_extra(self, logger, paths):
        """
        Log extra information per iteration based on the collected paths
        """
        pass


class DiscreteQFunction(QFunction):

    def get_qval_sym(self, obs_var, train=False):
        raise NotImplementedError


class ContinuousQFunction(QFunction):

    def get_qval_sym(self, obs_var, action_var, train=False):
        raise NotImplementedError

    def get_default_updates(self, obs_var, action_var, train=False):
        return {}


class NormalizableQFunction(QFunction):

    @property
    def normalizable(self):
        raise NotImplementedError

    def get_output_W(self):
        raise NotImplementedError

    def set_output_W(self, W_new):
        raise NotImplementedError

    def get_output_b(self):
        raise NotImplementedError

    def set_output_b(self, b_new):
        raise NotImplementedError
