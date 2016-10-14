class Terminator(object):
    """
    A terminator overrides the default termination condition of an environment.
    In general, it can take into account various training information. e.g.
    - whether the agent "gets stuck" (performing actions that lead to no state change)
    - whether training can be more sample efficient if the environment terminates and a new traj starts
    """
    def __init__(self,env):
        self.env = env

    def is_terminal(self):
        raise NotImplementedError
