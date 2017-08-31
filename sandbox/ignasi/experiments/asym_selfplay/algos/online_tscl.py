import numpy as np

class Online_TCSL:

    #TODO: Implement absolute value version
    def __init__(self, starts, eps = 0.1, alpha = 0.1, boltzmann = False, temperature = 0.0004):
        self.starts = starts # currently not used beyond getting
        self.num_starts = len(starts)
        self.q_vals = np.zeros(self.num_starts)
        self.eps = eps
        self.alpha = alpha
        self.reward_prev = np.zeros(self.num_starts)
        self.boltzmann = boltzmann
        if self.boltzmann:
            self.temperature = 0.0004

    def get_distribution(self, boltzmann = False):
        # If first step, return a uniform distribution
        if self.reward_prev is None:
            dist = np.ones(self.num_starts) / self.num_starts
            return dist

        if not self.boltzmann:
            dist = np.ones(self.num_starts) * (self.eps / self.num_starts) # can cache to not compute everytime
            dist[np.argmax(self.q_vals)] += (1 - self.eps)
            dist = np.array(dist)
            assert (abs(np.sum(dist) - 1) < 0.02)
            return dist / (np.sum(dist))
        else:
            dist = np.exp(self.q_vals / 0.0004)
            return dist / (np.sum(dist))

    def get_q(self):
        return self.q_vals

    def update_q(self, rewards, updated = None):
        # rewards is same length as starts
        # update is boolean array of whether or not we have enough trajectories to update a task

        improvement = rewards - self.reward_prev
        new_q_vals = self.alpha * improvement + (1 - self.alpha) * self.q_vals
        # a bit of a hack to only update q_vals if we have enough trajectories
        self.q_vals = updated * new_q_vals + (1 - updated) * self.q_vals
        self.reward_prev = updated * rewards + (1 - updated) * self.reward_prev


