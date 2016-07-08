import numpy as np

from rllab import spaces
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.misc import autoargs
from rllab.misc.overrides import overrides

BIG = 1e6

class FlexibleCartpoleEnv(CartpoleEnv):
    @autoargs.inherit(CartpoleEnv.__init__)
    def __init__(self, *args, **kwargs):
        super(FlexibleCartpoleEnv, self).__init__(*args, **kwargs)
        self.set_pole_size()

    @overrides
    def reset(self):
        return super(FlexibleCartpoleEnv, self).reset()

    @property
    @overrides
    def observation_space(self):
        if self.position_only:
            d = len(self._get_position_ids())
        else:
            d = len(self.extra_data.states)
        d += len(self._extra())
        ub = BIG * np.ones(d)
        return spaces.Box(ub*-1, ub)

    def get_raw_obs(self):
        raw_obs = super(FlexibleCartpoleEnv, self).get_raw_obs()
        return np.concatenate([raw_obs, self._extra()])

    def _extra(self):
        return [self._pole_w, self._pole_l]

    def set_pole_size(self, w=None, l=None):
        shape = self.pole.fixtures[0].shape
        w = w or shape.vertices[1][0]*2
        l = l or shape.vertices[1][1]
        shape.set_vertex(0, ( w/2,0))
        shape.set_vertex(1, ( w/2,l))
        shape.set_vertex(2, (-w/2,l))
        shape.set_vertex(3, (-w/2,0))
        self._pole_w, self._pole_l = w, l
