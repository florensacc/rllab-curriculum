from rllab.core.parameterized import Parameterized
from rllab.agent.base import Agent


class Policy(Parameterized, Agent):

    def __init__(self, env_spec):
        self._env_spec = env_spec

    # Should be implemented by all policies

    # def act(self, observation):
    #     """
    #     :param observation: The current observation
    #     :return: A pair (action, action_info), where action should be a member of the action space for the
    #     environment, and action_info should be a dictionary containing information about the distribution of actions
    #     """
    #     raise NotImplementedError

    @property
    def observation_space(self):
        return self._env_spec.observation_space

    @property
    def action_space(self):
        return self._env_spec.action_space

    @property
    def is_recurrent(self):
        return False


    def log_extra(self, paths):
        """
        Log extra information per iteration based on the collected paths
        """
        pass

    @property
    def recurrent(self):
        """
        Signals whether the policy is recurrent.
        """
        return False

    @property
    def info_keys(self):
        """
        List of keys that would be returned by calling get_action()
        :return:
        """
        return list()

    def info_sym(self, obs_var, action_var):
        """
        Return the distribution information about the actions given the observations (and possibly past actions,
        for recurrent policies).
        :return:
        """
        return dict()


class StochasticPolicy(Policy):
    def kl_sym(self, old_info_vars, new_info_vars):
        raise NotImplementedError

    def likelihood_ratio_sym(self, action_var, old_info_vars, new_info_vars):
        raise NotImplementedError

    def entropy(self, info):
        raise NotImplementedError

    def log_likelihood_sym(self, obs_var, action_var):
        raise NotImplementedError

    # def get_pdist_sym(self, obs_var, action_var):
    #     raise NotImplementedError

    # @property
    # def pdist_dim(self):
    #     raise NotImplementedError

    @property
    def dist_family(self):
        raise NotImplementedError
