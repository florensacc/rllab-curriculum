class VecEnv(object):
    """
    Define the abstract interface for vectorized env executors.
    """

    """
    Number of vectorized environments
    """
    n_envs = None

    def reset_trial(self, dones, seeds=None, *args, **kwargs):
        """
        Reset the specified environments, such that a new random MDP is drawn.
        :param dones: If provided, should be a numpy array of length == n_envs. By default it resets all environments.
        :param seeds: If provided, should be a numpy array of length == sum(dones) (i.e. only the relevant entries
        are provided). By default it uses the builtin seed
        :return: The initial observation only for the resetted environments. If none of the environments need reset,
        return an empty array instead.
        """
        raise NotImplementedError

    def reset(self, dones, seeds=None, *args, **kwargs):
        """
        Reset the specified environments while keeping the drawn MDP.
        :param dones: If provided, should be a numpy array of length == n_envs
        :param seeds: If provided, should be a numpy array of length == sum(dones) (i.e. only the relevant entries
        are provided)
        :return: The initial observation only for the resetted environments. If none of the environments need reset,
        return an empty array instead
        """
        raise NotImplementedError

    def terminate(self):
        """
        Clean up resources.
        """
        pass

    def step(self, action_n, max_path_length):
        """
        Advance all environments. This method should not handle resets of environments, but should indicate termination
        due to actual termination or maximum horizon is reached.
        :param action_n: A list of actions for each environment.
        :param max_path_length: Maximum horizon of execution.
        :return: A 4-tuple (next_obs, rewards, dones, infos). infos is provided as a single dictionary,
        where the entries are aggregated
        """
        raise NotImplementedError
