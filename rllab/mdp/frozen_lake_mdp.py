import numpy as np
from .base import MDP

from rllab.misc.overrides import overrides
from rllab.misc.serializable import Serializable
from rllab.misc import autoargs

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


MAPS = {
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ],
    "8x8":[
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ],
}


class FrozenLakeMDP(MDP, Serializable):
    """
    (Mostly copied from John's code)
    Winter is here. You and your friends were tossing around a frisbee at the park
    when you made a wild throw that left the frisbee out in the middle of the lake.
    The water is mostly frozen, but there are a few holes where the ice has melted.
    If you step into those holes, you'll fall into the freezing water.
    At this time, there's an international frisbee shortage, so it's absolutely imperative that
    you navigate across the lake and retrieve the disc.
    However, the ice is slippery, so you won't always move in the direction you intend.
    The episode ends when you reach the goal or fall in a hole.
    You receive a reward of 1 if you reach the goal, and zero otherwise.

    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your frozen doom
    G : goal, where the frisbee is located

    """

    @autoargs.arg("default_map", type=str, help="Choose from default maps, either 4x4 or 8x8")
    def __init__(self, desc=None, default_map='4x4', **kwargs):
        if desc is None and default_map is None:
            raise ValueError('Must provide either desc or default_map')
        elif desc is None:
            desc = MAPS[default_map]
        self.desc = np.array(map(lambda x: map(lambda c: c, x), desc))
        nrow, ncol = self.desc.shape
        self.maxxy = np.array([nrow-1, ncol-1])
        (startx,), (starty,) = np.nonzero(self.desc=='S')
        self.startstate = np.array([startx,starty])
        Serializable.__init__(self, desc)

    def step(self, state, action):

        action = (action + np.random.randint(-1, 2)) % 4
        increments = np.array([[0,-1],[1,0],[0,1],[-1,0]])
        nextstate = np.clip(state + increments[action], [0,0], self.maxxy)
        statetype = self.desc[nextstate[0],nextstate[1]]

        holemask = statetype == 'H'
        goalmask = statetype == 'G'
        if goalmask or holemask:
            nextstate = self.startstate
            done = True
        else:
            done = False
        reward = 1 if goalmask else 0
        return nextstate, self._get_observation(nextstate), reward, done

    @property
    @overrides
    def action_dim(self):
        return 4

    @property
    def observation_shape(self):
        nrow, ncol = self.desc.shape
        return (nrow*ncol,)

    def reset(self):
        state = np.copy(self.startstate)
        return state, self._get_observation(state)

    def _get_observation(self, state):
        nrow, ncol = self.desc.shape
        return np.eye(nrow*ncol)[state[0]*ncol+state[1]]
