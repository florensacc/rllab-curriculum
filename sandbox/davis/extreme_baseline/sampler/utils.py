import numpy as np
from rllab.misc import tensor_utils
import time
from rllab.envs.normalized_env import NormalizedEnv


def rollout(env, agent, max_path_length=np.inf, animated=False, speedup=1):
    env_has_get_state = isinstance(env, NormalizedEnv) and hasattr(env._wrapped_env, 'get_state') \
        or not isinstance(env, NormalizedEnv) and hasattr(env, 'get_state')

    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    env_states = []
    o = env.reset()
    agent.reset()
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        if env_has_get_state:
            env_states.append(env.get_state())
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

    return dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
        env_states=tensor_utils.stack_tensor_list(env_states),
    )
