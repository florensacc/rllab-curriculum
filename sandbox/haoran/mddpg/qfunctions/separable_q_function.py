import abc


class SeparableQFunction(object):
    """
    A Q-function that's split up into

    Q(state, action) = A(state, action) + V(state)
    """
    @abc.abstractmethod
    def get_implicit_value_function(self):
        return

    @abc.abstractmethod
    def get_implicit_advantage_function(self):
        return