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

    def reset(self):
        """

        :return:
        """
        # This is a dummy method that allows for
        # random initializations before each episode.
        # A potential usage is for mixture or recurrent policies, where one
        # of the mixture distributions is selected at the beginning
        # of each episode
        pass
