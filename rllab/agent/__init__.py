class Agent(object):
    """
    Base class for the agent in a reinforcement learning problem

    By convention, the constructor of a subclass should have the form
    __init__(self, observation_space, action_space, ...):

    Also by convention, you should `raise error.UnsupportedSpace` when passed
    a Space that you're not compatible with.
    """

    def act(self, observation):
        """
        Input
        -----
        observation : provided by the environment

        Outputs
        -------
        (action, info)

        action : array
        info : dictionary containing other diagnostic information
        """
        raise NotImplementedError
