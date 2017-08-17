import numpy as np
from rllab.misc import tensor_utils
import time


def rollout(env, agent, max_path_length=np.inf, animated=False, speedup=1, init_state=None, no_action = False):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    dones = []
    # no_action = True
    if init_state is not None:
        o = env.reset(init_state)
    else:
        o = env.reset()
    agent.reset()
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        if no_action:
            a = np.zeros_like(a)
        next_o, r, d, env_info = env.step(a)
        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        dones.append(d)
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)
    if animated:
        env.render(close=False)

    return dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
        dones=np.asarray(dones),
        last_obs=o,
    )
