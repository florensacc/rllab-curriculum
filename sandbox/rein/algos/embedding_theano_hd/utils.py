import numpy as np
from rllab.misc import tensor_utils
import time

print('NOTE: n_seq_frames=4 hardcoded!')
def rollout(env, agent, max_path_length=np.inf, animated=False, speedup=1, n_seq_frames=4):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    path_length = 0
    if animated:
        env.render()
    # buffer of empty frames.
    top_ptr = 0
    obs_bfr = np.zeros((n_seq_frames, env.spec.observation_space.shape[2], env.spec.observation_space.shape[1]))
    # We fill up with the starting frame, better than black.
    for i in range(n_seq_frames):
        obs_bfr[i] = o
    while path_length < max_path_length:
        obs_bfr[top_ptr] = o
        # last frame [-1]
        _o = np.concatenate((obs_bfr[(top_ptr + 1) % n_seq_frames:],
                            obs_bfr[:(top_ptr + 1) % n_seq_frames]), axis=0)
        # format for policy/baseline is w x h x n_samp
        # o = o.transpose((2, 1, 0))
        a, agent_info = agent.get_action(_o)
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
        top_ptr = (top_ptr + 1) % n_seq_frames
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
