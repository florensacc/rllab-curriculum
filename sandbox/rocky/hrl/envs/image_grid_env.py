

from rllab.envs.base import Step
import numpy as np

from rllab.envs.base import Env
from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
from rllab.spaces.discrete import Discrete

AGENT = 0
GOAL = 1
N_OBJECT_TYPES = 2


class ImageGridEnv(Env, Serializable):
    """
    A simple grid world environment, environment with a grid world Seaquest-like grid world game.

    The observation is a 3D array where the last two dimensions encode the coordinate and the first dimension
    encode whether each type of objects is present in this grid.
    """

    def __init__(self, size=10, subgoal_interval=1, action_interval=1):
        """
        :param size: Size of the grid world
        :param subgoal_interval: if provided, will make sure that the goal is placed at a position where the
        coordinates are integer multiples of the subgoal interval
        :param action_interval: if provided, the number of actions of the environment will be (action_interval+1)**2,
        where each action corresponds to a sequence of actions that reach a particular state feasible in exactly
        action_interval steps.
        """
        Serializable.quick_init(self, locals())
        assert size / subgoal_interval >= 2, "Size of the grid world must be at least twice the subgoal interval"
        self.size = size
        self.subgoal_interval = subgoal_interval
        self.action_interval = action_interval
        self.goal_pos = None
        self.agent_pos = None
        self._observation_space = Box(low=0, high=N_OBJECT_TYPES, shape=(N_OBJECT_TYPES, self.size, self.size))
        self._feasible_actions = self._gen_feasible_actions(action_interval)
        self._action_space = Discrete(len(self._feasible_actions))

    def _gen_feasible_actions(self, action_interval):
        """
        Generate the list of feasible actions, given the action interval
        """
        k = action_interval

        actions = []

        for k in range(action_interval, -1, -2):
            for inc in range(k + 1):
                actions.append((k - inc, inc))
                actions.append((k - inc, -inc))
                actions.append((inc - k, -inc))
                actions.append((inc - k, inc))
        actions = sorted(set(actions))
        return actions

    def reset(self):
        self.agent_pos = (0, 0)
        self.goal_pos = self.agent_pos
        while self.goal_pos == self.agent_pos:
            self.goal_pos = tuple(
                np.random.randint(low=0, high=self.size / self.subgoal_interval, size=2) * self.subgoal_interval
            )
        # always start at the top left corner
        return self.get_current_obs()

    def get_current_obs(self):
        grid = np.zeros((N_OBJECT_TYPES, self.size, self.size), dtype='uint8')
        grid[(AGENT,) + self.agent_pos] = 1
        grid[(GOAL,) + self.goal_pos] = 1
        return grid

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def step(self, action):
        coords = np.array(self.agent_pos)
        # increments = np.array([[-1, 0], [0, -1], [0, 1], [1, 0]])
        self.agent_pos = tuple(np.clip(
            coords + self._feasible_actions[action],#increments[action],
            [0, 0],
            [self.size - 1, self.size - 1]
        ))
        if self.agent_pos == self.goal_pos:
            return Step(observation=self.get_current_obs(), reward=1, done=True)
        return Step(observation=self.get_current_obs(), reward=0, done=False)

    @staticmethod
    def action_from_direction(d):
        """
        Return the action corresponding to the given direction. This is a helper method for debugging and testing
        purposes.
        :return: the action index corresponding to the given direction
        """
        return dict(
            top=0,
            left=1,
            right=2,
            bottom=3
        )[d]
