from __future__ import print_function
from __future__ import absolute_import
from rllab.misc import tensor_utils
import numpy as np
import random
from contextlib import contextmanager


def downsample_tensor_dict(tensor_dict, interval):
    """
    Given a dictionary of (tensors or dictionary of tensors), downsample each of the tensors according to the interval
    :param tensor_dict: A dictionary of (tensors or dictionary of tensors)
    :return: a dictionary with downsampled tensors
    """
    ret = dict()
    for k, v in tensor_dict.iteritems():
        if isinstance(v, dict):
            ret[k] = downsample_tensor_dict(v, interval=interval)
        else:
            ret[k] = v[::interval]
    return ret


def downsample_path(path, downsample_interval):
    rewards = path['rewards']
    path_length = len(rewards)
    chunked_length = int(np.ceil(path_length * 1.0 / downsample_interval))
    padded_length = chunked_length * downsample_interval
    padded_rewards = np.append(rewards, np.zeros(padded_length - path_length))
    chunked_rewards = np.sum(
        np.reshape(padded_rewards, (chunked_length, downsample_interval)),
        axis=1
    )

    new_dict = dict()
    new_dict["rewards"] = chunked_rewards

    for k, val in path.iteritems():
        if k == "rewards":
            pass
        elif isinstance(val, dict):
            new_dict[k] = downsample_tensor_dict(val, downsample_interval)
        else:
            new_dict[k] = val[::downsample_interval]

    return new_dict


@contextmanager
def using_seed(seed):
    assert seed is not None
    npr_state = np.random.get_state()
    random_state = random.getstate()
    np.random.seed(seed)
    random.seed(seed)
    yield
    np.random.set_state(npr_state)
    random.setstate(random_state)
