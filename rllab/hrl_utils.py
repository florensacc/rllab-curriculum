from __future__ import print_function
from __future__ import absolute_import
from rllab.misc import tensor_utils
import numpy as np


def subsample_path(path, subsample_interval):
    rewards = path['rewards']
    path_length = len(rewards)
    chunked_length = int(np.ceil(path_length * 1.0 / subsample_interval))
    padded_length = chunked_length * subsample_interval
    padded_rewards = np.append(rewards, np.zeros(padded_length - path_length))
    chunked_rewards = np.sum(
        np.reshape(padded_rewards, (chunked_length, subsample_interval)),
        axis=1
    )

    new_dict = dict()
    new_dict["rewards"] = chunked_rewards

    for k, val in path.iteritems():
        if k == "reward":
            pass
        elif isinstance(val, dict):
            new_dict[k] = tensor_utils.subsample_tensor_dict(val, subsample_interval)
        else:
            new_dict[k] = val[::subsample_interval]

    return new_dict


    # observations = path['observations']
    # actions = path['actions']
    #
    # chunked_env_infos = tensor_utils.subsample_tensor_dict(path["env_infos"], subsample_interval)
    # chunked_agent_infos = tensor_utils.subsample_tensor_dict(path["agent_infos"], subsample_interval)
    # chunked_actions = actions[::subsample_interval]
    # chunked_observations = observations[::subsample_interval]
    # return dict(
    #     observations=chunked_observations,
    #     actions=chunked_actions,
    #     env_infos=chunked_env_infos,
    #     agent_infos=chunked_agent_infos,
    #     rewards=chunked_rewards,
    # )
