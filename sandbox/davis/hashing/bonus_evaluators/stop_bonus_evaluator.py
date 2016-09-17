import numpy as np
from sandbox.davis.hashing.bonus_evaluators.hashing_bonus_evaluator import HashingBonusEvaluator


class StopBonusEvaluator(HashingBonusEvaluator):
    def __init__(self, env_spec, **kwargs):
        super(StopBonusEvaluator, self).__init__(env_spec, **kwargs)
        self.stopped = False

    def fit_before_process_samples(self, paths):
        rewards = np.concatenate([p["rewards"] for p in paths])
        if np.max(rewards) > 0:
            self.stopped = True
        super(StopBonusEvaluator, self).fit_before_process_samples(paths)

    def predict(self, paths):
        return (1 - self.stopped) * super(StopBonusEvaluator, self).predict(paths)
