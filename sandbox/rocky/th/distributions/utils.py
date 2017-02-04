import rllab.spaces as spaces
import sandbox.rocky.tf.spaces as tf_spaces
import numpy as np
import torch.nn.functional as F
import itertools

from sandbox.rocky.th.distributions.categorical import Categorical
from sandbox.rocky.th.distributions.product_distribution import ProductDistribution


def space_to_dist_dim(space):
    if isinstance(space, (spaces.Discrete, tf_spaces.Discrete)):
        return space.n
    elif isinstance(space, (spaces.Box, tf_spaces.Box)):
        return space.flat_dim * 2
    elif isinstance(space, (spaces.Product, tf_spaces.Product)):
        components = space.components
        return sum(map(space_to_dist_dim, components))
    else:
        raise NotImplementedError


def output_to_info(output, output_space):
    if isinstance(output_space, (spaces.Discrete, tf_spaces.Discrete)):
        return dict(prob=F.softmax(output))
    elif isinstance(output_space, (spaces.Box, tf_spaces.Box)):
        raise NotImplementedError
        # mean = output_var[:, :output_space.flat_dim]
        # log_std = output_var[:, output_space.flat_dim:]
        # return dict(mean=mean, log_std=log_std)
    elif isinstance(output_space, (spaces.Product, tf_spaces.Product)):
        components = output_space.components
        dimensions = [space_to_dist_dim(x) for x in components]
        cum_dims = list(np.cumsum(dimensions))
        ret = dict()
        for idx, slice_from, slice_to, subspace in zip(itertools.count(), [0] + cum_dims, cum_dims, components):
            sub_info = output_to_info(output[:, slice_from:slice_to], subspace)
            for k, v in sub_info.items():
                ret["id_%d_%s" % (idx, k)] = v
        return ret
    else:
        raise NotImplementedError


def space_to_distribution(space):
    """
    Build a distribution from the given space
    """
    if isinstance(space, (spaces.Discrete, tf_spaces.Discrete)):
        return Categorical(space.n)
    elif isinstance(space, (spaces.Box, tf_spaces.Box)):
        raise NotImplementedError
    elif isinstance(space, (spaces.Product, tf_spaces.Product)):
        components = space.components
        component_dists = list(map(space_to_distribution, components))
        return ProductDistribution(component_dists)
    else:
        raise NotImplementedError
