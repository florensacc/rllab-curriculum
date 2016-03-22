from .env_spec import EnvSpec


class Env(object):
    """
    Adopted from rl-gym, OpenAI.
    """

    def step(self, action):
        """
        Run one timestep of the environment's dynamics

        Input
        -----
        action : an array provided by the agent

        Outputs
        -------
        (observation, reward, done, info), a namedtuple

        observation : array
        reward : a float scalar
        done : a boolean, indicating whether the episode has ended
        info : a dictionary containing other diagnostic information from the step.
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset the state of the environment.
        Returns the initial observation.
        """
        raise NotImplementedError

    @property
    def action_space(self):
        """
        Returns a Space object
        """
        raise NotImplementedError

    @property
    def observation_space(self):
        """
        Returns a Space object
        """
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def log_extra(self, paths):
        pass

    @property
    def spec(self):
        return EnvSpec(
            observation_space=self.observation_space,
            action_space=self.action_space,
        )
