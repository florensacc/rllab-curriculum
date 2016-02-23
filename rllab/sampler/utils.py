import numpy as np
import pyprind


def rollout(mdp, policy, max_length=np.inf, animated=False, speedup=1):
    observations = []
    actions = []
    rewards = []
    pdists = []
    o = mdp.reset()
    policy.episode_reset()
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
        self.pbar = pyprind.ProgBar(self.max_progress)

    def inc(self, increment):
        self.cur_count += increment
        new_progress = self.cur_count * self.max_progress / self.total_count
        if new_progress < self.max_progress:
            self.pbar.update(new_progress - self.cur_progress)
        self.cur_progress = new_progress

    def stop(self):
        self.pbar.stop()
