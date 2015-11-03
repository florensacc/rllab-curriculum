import numpy as np
import pyprind

def rollout(mdp, policy, max_length=np.inf, animated=False, use_state=False):
    states = []
    observations = []
    actions = []
    rewards = []
    pdists = []
    s, o = mdp.reset()
    path_length = 0
    while path_length < max_length:
        if use_state:
            a, pdist = policy.get_action(s, path_length)
        else:
            a, pdist = policy.get_action(o)
        print a
        next_s, next_o, r, d = mdp.step(s, a)
        states.append(s)
        observations.append(o)
        rewards.append(r)
        actions.append(a)
        pdists.append(pdist)
        path_length += 1
        if d:
            break
        s, o = next_s, next_o
        if animated:
            mdp.plot()
            #import time
            #time.sleep(1)
    return dict(
        states=np.vstack(states),
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
