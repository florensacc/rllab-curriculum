import numpy as np
from .base import MDP

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class FrozenLakeMDP(MDP):
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

    def __init__(self, desc):
        self.desc = np.asarray(desc,dtype='c1')
        nrow, ncol = self.desc.shape
        self.maxxy = np.array([nrow-1, ncol-1])
        (startx,), (starty,) = np.nonzero(self.desc=='S')
        self.startstate = np.array([startx,starty])

    def step(self, states, actions):
        n = len(states)
        actions = actions[0]
        assert len(actions)==n
        actions = (actions + np.random.randint(-1,2,len(actions))) % 4
        increments = np.array([[0,-1],[1,0],[0,1],[-1,0]])
        nextstates = np.clip(states + increments[actions], [0,0], self.maxxy)
        rewards = np.zeros((n, 1))
        dones = np.zeros((n,),bool)
        statetype = self.desc[nextstates[:,0],nextstates[:,1]]

        holemask = statetype == 'H'
        goalmask = statetype == 'G'
        nextstates[goalmask | holemask] = self.startstate
        dones[goalmask | holemask] = True
        rewards[goalmask] = 1
        return nextstates, nextstates, rewards, dones, [1] * n

    @property
    def action_set(self):
        return [[0, 1, 2, 3]]

    @property
    def action_dims(self):
        return [4]

    @property
    def observation_shape(self):
        return (2,)

    def sample_initial_states(self, n):
        s = np.zeros((n,2),'i')+self.startstate[None,:]
        return s,s
