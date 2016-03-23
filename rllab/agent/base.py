class Agent(object):
    """
    Base class for the agent in a reinforcement learning problem
    By convention, the constructor of a subclass should have the form
    __init__(self, observation_space, action_space, ...):
    Also by convention, you should `raise error.UnsupportedSpace` when passed
    a Space that you're not compatible with.
    """

    def act(self, observation, new, reward):
        """
        Processes a response from the environment.
        Input
        -----
        observation : observation from the environment
        new : a boolean value indicating whether the episode has reset
        reward : reward from the previous step of the environment
        Outputs
        -------
        (action, info)
        action : the action taken by the agent after receiving the new observation
        info : dictionary containing other diagnostic information
        """
        raise NotImplementedError
