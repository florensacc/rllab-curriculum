import numpy as np
from rllab.misc import tensor_utils
import time
from rllab.envs.normalized_env import NormalizedEnv  # this is just to check if the env passed is a normalized maze


def rollout(env, agent, max_path_length=np.inf, reset_start_rollout=False, animated=False, speedup=1):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    if reset_start_rollout:
        o = env.reset()  # otherwise it will never advance!!
    else:
        if isinstance(env, NormalizedEnv):
            o = env.wrapped_env.get_current_obs()
        else:
            o = env.get_current_obs()
    agent.reset()
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        # print("obs {} is {}".format(path_length, o))
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)
    # if animated:
        # env.render(close=True)

    return dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),  # here it concatenates all lower-level paths!
        #  So all elements are np.arrays of max_path_length x time_steps_agg x corresp_dim
        #  hence the next concatenation done by sampler at the higher level doesn't work because the mismatched dim
        #  1 and not 0!!
    )
