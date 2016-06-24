class VecEnv(object):
    """
    Vectorized environment base class
    """
    def step(self, vac):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)

        where 'news' is a boolean vector indicating whether each element is new.
        """
        raise NotImplementedError
    def reset(self):
        """
        Reset all environments
        """
        raise NotImplementedError
    @property
    def action_space(self):
        raise NotImplementedError
    @property
    def observation_space(self):
        return self._observation_space
    
    