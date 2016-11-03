from sandbox.davis.hashing.bonus_evaluators.hashing_bonus_evaluator import HashingBonusEvaluator
from rllab.spaces.box import Box
from rllab.envs.env_spec import EnvSpec

import numpy as np

BIG = 1e6

X = 0
Z = 1
Y_angle = 2
X_vel = 9
Z_vel = 10
Y_ang_vel = 11


class HalfCheetahHackyHash(HashingBonusEvaluator):
    def __init__(self, env_spec, indices=[X, Z], *args, **kwargs):
        upper_bound = BIG * np.ones(2)
        obs_space = Box(-1 * upper_bound, upper_bound)
        env_spec = EnvSpec(obs_space, env_spec.action_space)
        super(HalfCheetahHackyHash, self).__init__(env_spec, *args, **kwargs)
        self.indices = indices

    def fit_before_process_samples(self, paths):
        # import pdb; pdb.set_trace()
        paths = [self.grab_position_indices(path) for path in paths]
        super(HalfCheetahHackyHash, self).fit_before_process_samples(paths)

    def predict(self, path):
        path = self.grab_position_indices(path)
        return super(HalfCheetahHackyHash, self).predict(path)

    def grab_position_indices(self, path):
        import pdb; pdb.set_trace()
        return {"observations": path["observations"][:, self.indices]}
