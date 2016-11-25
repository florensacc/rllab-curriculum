import abc


class OptimizableQFunction(object):
    """
    A Q-function that implicitly has a policy.
    """
    @abc.abstractmethod
    def get_implicit_policy(self):
        return