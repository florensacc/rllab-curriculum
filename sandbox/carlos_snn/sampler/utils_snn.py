import numpy as np
import time
from rllab.misc import tensor_utils
from rllab.envs.normalized_env import NormalizedEnv  # to check if env is normalized => no method get_current_obs()


def rollout_snn(env, agent, max_path_length=np.inf, reset_start_rollout=True,
                switch_lat_every=0, animated=False, speedup=1):
    """
    :param reset_start_rollout: whether to reset at the start of every rollout
    :param switch_lat_every: potential change in latents (by resetting the agent with forced resample lat)
    """
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    if reset_start_rollout:
        o = env.reset()  # in case rollout is called to produce parts of a trajectory: otherwise it will never advance!!
    else:
        if isinstance(env, NormalizedEnv):
            o = env.wrapped_env.get_current_obs()
        else:
            o = env.get_current_obs()
    agent.reset()  # this resamples a latent in SNNs!
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        if switch_lat_every > 0 and path_length % switch_lat_every == 0:
            agent.reset(force_resample_lat=True)  # here forced to resample a latent
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
