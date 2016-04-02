from rllab.core.parameterized import Parameterized


class Policy(Parameterized):
    def __init__(self, env_spec):
        self._env_spec = env_spec

    # Should be implemented by all policies

    def get_action(self, observation):
        raise NotImplementedError

    def reset(self):
        pass

    @property
    def observation_space(self):
        return self._env_spec.observation_space

    @property
    def action_space(self):
        return self._env_spec.action_space

    @property
    def recurrent(self):
        """
        Indicates whether the policy is recurrent.
        :return:
        """
        return False

    def log_diagnostics(self, paths):
        """
        Log extra information per iteration based on the collected paths
        """
        pass


class StochasticPolicy(Policy):
    # def kl_sym(self, old_dist_info_vars, new_dist_info_vars):
    #     return self.dist_family.kl_sym(old_dist_info_vars)
    #     raise NotImplementedError
    #
    # def likelihood_ratio_sym(self, action_var, old_dist_info_vars, new_dist_info_vars):
    #     raise NotImplementedError

    # def entropy(self, dist_info):
    #     raise NotImplementedError

    # def log_likelihood_sym(self, obs_var, action_var):
    #     raise NotImplementedError

    @property
    def distribution(self):
        """
        :rtype Distribution
        """
        raise NotImplementedError

    # @property
    # def dist_info_keys(self):
    #     """
    #     List of keys in the agent_info object related to information about the action distribution given the
    #     observations
    #     :return:
    #     """
    #     return list()

    def dist_info_sym(self, obs_var, action_var):
        """
        Return the symbolic distribution information about the actions.
        :return:
        """
        raise NotImplementedError

    def dist_info(self, obs, actions):
        """
        Return the distribution information about the actions.
        :return:
        """
        raise NotImplementedError