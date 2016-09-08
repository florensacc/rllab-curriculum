""" This file defines the base trajectory optimization class. """
import abc


class TrajOpt(object, metaclass=abc.ABCMeta):
    """ Trajectory optimization superclass. """

    def __init__(self, hyperparams):
        self._hyperparams = hyperparams

    @abc.abstractmethod
    def update(self):
        """ Update trajectory distributions. """
        raise NotImplementedError("Must be implemented in subclass.")


# TODO - Interface with C++ traj opt?
