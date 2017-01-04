from rllab.misc import logger
from collections import deque
import numpy as np

class PrioritizedPathReplayer(object):
    """
    Assign priority scores to paths based on raw returns and replay paths with probability determined by the scores.
    Not bothering to use importance sampling.
    Other design choices
    - add importance sampling to account for bias
    - replay different time steps rather than the entire traj
    - disallow bonus_evaluator from using the replayer
    """

    def __init__(
        self,
        alpha=100,
        use_raw_return=True,
        n_replay=None,
        max_recorded_paths=5,
        record_zero_return_paths=False,
        random_replay=False,
    ):
        """
        :param n_replay: number of paths replayed; None means same as the total number of recorded paths
        """
        self.alpha = alpha
        self.recorded_paths = []
        self.n_replay = n_replay
        self.recorded_paths = deque(maxlen=max_recorded_paths)
        self.record_zero_return_paths = record_zero_return_paths
        self.random_replay = random_replay
        self.use_raw_return = use_raw_return

    def retrieve_return(self, path):
        if self.use_raw_return:
            return np.sum(path["raw_rewards"])
        else:
            return path["returns"][0]

    def compute_scores(self, paths):
        returns = np.asarray([self.retrieve_return(path) for path in paths])
        scores = np.exp(self.alpha * returns)
        return scores

    def replay_paths(self):
        if self.random_replay:
            scores = self.compute_scores(self.recorded_paths)
            probs = scores / np.sum(scores)

            if self.n_replay is not None:
                n_replay = self.n_replay
            else:
                n_replay = len(self.recorded_paths)

            def draw_from_probs(probs):
                return list(np.random.multinomial(1,probs)).index(1)
            path_indices = [draw_from_probs(probs) for _ in range(n_replay)]
            paths = [self.recorded_paths[i] for i in path_indices]
        else:
            paths = list(self.recorded_paths)

        return paths

    def record_paths(self,paths):
        if not self.record_zero_return_paths:
            returns = np.asarray([self.retrieve_return(path) for path in paths])
            for path, R in zip(paths, returns):
                if R > 0:
                    self.recorded_paths.append(path)
        else:
            for path in paths:
                self.recorded_paths.append(path)

        self.log("total paths recorded: %d"%(len(self.recorded_paths)))

    def log(self,message):
        logger.log("path replayer: " + message)
