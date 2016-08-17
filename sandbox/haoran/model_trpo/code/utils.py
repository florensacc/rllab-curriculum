def central_finite_difference_jacobian(func,inputs,output_dim,step,wrt=0):
    n = len(inputs[wrt])
    m = output_dim
    J = np.zeros((m,n))
    def f(wrt_input):
        _inputs = inputs
        _inputs[wrt] = wrt_input
        return func(*_inputs)

    wrt_input = inputs[wrt]
    for i in range(n):
        d_wrt_input = np.zeros(n)
        d_wrt_input[i] = step
        output_plus = f(wrt_input + d_wrt_input)
        output_minus = f(wrt_input - d_wrt_input)
        J[:,i] = (output_plus - output_minus).ravel() / (2. * step)
    return J


import numpy as np
from rllab.misc import tensor_utils
import time

def rollout(env, agent, max_path_length=np.inf, animated=False, speedup=1,record_full_state=False):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    if record_full_state:
        s = env._full_state
    agent.reset()
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        if record_full_state:
            env_info["full_states"] = s
            s = env._full_state
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
    )
