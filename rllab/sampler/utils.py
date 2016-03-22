import numpy as np
import pyprind
from rllab.misc import logger
from rllab.misc import tensor_utils


def rollout(env, agent, max_length=np.inf, animated=False, speedup=1):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    path_length = 0
    while path_length < max_length:
        a, agent_info = agent.act(o)
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
    return dict(
        observations=tensor_utils.stack_tensors(observations),
        actions=tensor_utils.stack_tensors(actions),
        rewards=tensor_utils.stack_tensors(rewards),
        agent_infos=tensor_utils.stack_tensor_dicts(agent_infos),
        env_infos=tensor_utils.stack_tensor_dicts(env_infos),
    )


class ProgBarCounter(object):

    def __init__(self, total_count):
        self.total_count = total_count
        self.max_progress = 1000000
        self.cur_progress = 0
        self.cur_count = 0
        if not logger.get_log_tabular_only():
            self.pbar = pyprind.ProgBar(self.max_progress)
        else:
            self.pbar = None

    def inc(self, increment):
        if not logger.get_log_tabular_only():
            self.cur_count += increment
            new_progress = self.cur_count * self.max_progress / self.total_count
            if new_progress < self.max_progress:
                self.pbar.update(new_progress - self.cur_progress)
            self.cur_progress = new_progress

    def stop(self):
        if not logger.get_log_tabular_only():
            self.pbar.stop()
