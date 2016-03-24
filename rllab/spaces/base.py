import numpy as np


class Space(object):
    """
    Provides a classification state spaces and action spaces,
    so you can write generic code that applies to any Environment.
    E.g. to choose a random action.

    Adopted from rl-gym, OpenAI
    """

    def sample(self, seed=0):
        """
        Uniformly randomly sample a random elemnt of this space
        """
        raise NotImplementedError

    def contains(self, x):
        """
        Return boolean specifying if x is a valid
        member of this space
        """
        raise NotImplementedError

    def flatten(self, x):
        raise NotImplementedError

    def unflatten(self, x):
        raise NotImplementedError


    # def from_tensors(self, x):
    #     """
    #     Given a member of the space, convert it to a tensor representation
    #     :param x: a member of the space
    #     :return: a flat vector representation of x
    #     """
    #     raise NotImplementedError
    #
    # @property
    # def tensor_shapes(self):
    #     """
    #     The shape(s) of the tensor representation. A valid return value is either a single tuple, or a list of valid
    #     return values
    #     :return: the shape(s) of the tensor representation
    #     """
    #     raise NotImplementedError

    @property
    def flat_dim(self):
        """
        The dimension of the flattened vector of the tensor representation
        """
        raise NotImplementedError

    #
    # def untensorize(self, x):
    #     """
    #     Given a numpy tensor, convert it to a ordinary member of the space
    #     :param x: a flattened vector
    #     :return: a member of the space corresponding to x
    #     """
    #     raise NotImplementedError

    def new_tensor_variables(self, name, extra_dims):
        """
        Create one or a group of Theano tensor variables given the name and extra dimensions prepended
        :param name: name of the variable (or prefix, if the returned value is a group of variables)
        :param extra_dims: extra dimensions in the front
        :return: the created tensor variable(s)
        """
        raise NotImplementedError
