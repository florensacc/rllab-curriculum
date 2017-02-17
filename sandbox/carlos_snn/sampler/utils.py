import numpy as np
import time
from rllab.misc import tensor_utils
from rllab.envs.normalized_env import NormalizedEnv  # this is just to check if the env passed is a normalized maze


def rollout(env, agent, max_path_length=np.inf, reset_start_rollout=True, keep_rendered_rgbs=False,
            animated=False, speedup=1):
    """
    :param reset_start_rollout: whether to reset the env when calling this function
    :param keep_rendered_rgbs: whether to keep a list of all rgb_arrays (for future video making)
    """
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    terminated = []
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
    if keep_rendered_rgbs:  # will return a new entry to the path dict with all rendered images
        rendered_rgbs = [env.render(mode='rgb_array')]
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            terminated.append(1)
            break
        terminated.append(0)
        o = next_o
        if keep_rendered_rgbs:  # will return a new entry to the path dict with all rendered images
            rendered_rgbs.append(env.render(mode='rgb_array'))
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)
    # if animated:   # this is off as in the case of being an inner rollout, it will close the outer renderer!
        # env.render(close=True)

    path_dict = dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),  # here it concatenates all lower-level paths!
        # termination indicates if the rollout was terminated or if we simply reached the limit of steps: important
        # when BOTH happend at the same time, to still be able to know it was the done (for hierarchized envs)
        terminated=tensor_utils.stack_tensor_list(terminated),
    )
    if keep_rendered_rgbs:
        path_dict['rendered_rgbs'] = tensor_utils.stack_tensor_list(rendered_rgbs)

    return path_dict
