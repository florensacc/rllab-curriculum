from rllab.misc import autoargs
from rllab.misc.ext import new_tensor
from rllab.core.parameterized import Parameterized
import theano
import theano.tensor as TT


class Policy(Parameterized):

    def __init__(self, mdp_spec):
        self._mdp_spec = mdp_spec

    # Should be implemented by all policies

    def get_actions(self, observations):
        raise NotImplementedError

    def get_action(self, observation):
        actions, pdists = self.get_actions([observation])
        return actions[0], pdists[0]

    @property
    def observation_shape(self):
        return self._mdp_spec.observation_shape

    @property
    def observation_dim(self):
        return self._mdp_spec.observation_dim

    @property
    def observation_dtype(self):
        return self._mdp_spec.observation_dtype

    @property
    def action_dim(self):
        return self._mdp_spec.action_dim

    @property
    def action_dtype(self):
        return self._mdp_spec.action_dtype

    @property
    def is_recurrent(self):
        return False

    @classmethod
    @autoargs.add_args
    def add_args(cls, parser):
        pass

    @classmethod
    @autoargs.new_from_args
    def new_from_args(cls, args, mdp):
        pass

    def reset(self):
        # This is a dummy method that allows for
        # random initializations before each episode.
        # A potential usage is for mixture policies, where one
        # of the mixture distributions is selected at the beginning
        # of each episode
        pass

    def log_extra(self, paths):
        """
        Log extra information per iteration based on the collected paths
        """
        pass

    @property
    def recurrent(self):
        """
        Signals whether the policy is recurrent.
        If this returns true, get_action should take in the previous action as
        input, and some other methods might also behave differently
        """
        return False


class StochasticPolicy(Policy):

    def __init__(self, mdp):
        super(StochasticPolicy, self).__init__(mdp)
        self._f_log_prob = None

    def kl(self, old_pdist_var, new_pdist_var):
        raise NotImplementedError

    def likelihood_ratio(self, old_pdist_var, new_pdist_var, action_var):
        raise NotImplementedError

    def compute_entropy(self, pdist):
        raise NotImplementedError

    # Only needed for vanilla policy gradient & guided policy search
    def get_log_prob(self, observations, actions):
        if self._f_log_prob is None:
            input_var = new_tensor(
                'input',
                ndim=len(self.observation_shape) + 1,
                dtype=self.observation_dtype
            )
            action_var = TT.matrix('actions', dtype=self.action_dtype)
            self._f_log_prob = theano.function(
                [input_var, action_var],
                self.get_log_prob_sym(input_var, action_var),
                allow_input_downcast=True,
                on_unused_input='ignore'
            )
        return self._f_log_prob(observations, actions)

    def get_log_prob_sym(self, obs_var, action_var):
        raise NotImplementedError

    def get_pdist_sym(self, obs_var, action_var):
        raise NotImplementedError

    @property
    def pdist_dim(self):
        raise NotImplementedError
