import numpy as np
import pyprind
from rllab.misc import logger


def rollout(mdp, policy, max_length=np.inf, animated=False, speedup=1):
    observations = []
    actions = []
    rewards = []
    pdists = []
    o = mdp.reset()
    policy.reset()
    path_length = 0
    while path_length < max_length:
        a, pdist = policy.get_action(o)
        next_o, r, d = mdp.step(a)
        observations.append(o)
        rewards.append(r)
        actions.append(a)
        pdists.append(pdist)
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            mdp.plot()
            import time
            time.sleep(mdp.timestep / speedup)
    return dict(
        observations=np.vstack(observations),
        actions=np.vstack(actions),
        rewards=np.vstack(rewards).reshape(-1),
        pdists=np.vstack(pdists)
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
