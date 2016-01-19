#
# Copyright (C) 2013, Will Dabney
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np

from rllab.mdp.base import MDP
from rllab.misc.overrides import overrides
from rllab.misc import autoargs


class MountainCarND(object):
    """A generalized Mountain Car domain, which allows N-dimensional
    movement. When dimension=2 this behaves exactly as the classical
    Mountain Car domain. For dimension=3 it behaves as given in the
    paper:

    Autonomous Transfer for Reinforcement Learning. 2008.
    Matthew Taylor, Gregory Kuhlmann, and Peter Stone.

    However, this class also allows for even greater dimensions.
    """
    name = "3D Mountain Car"

    def __init__(self, **kwargs):
        dim = int(max(2, kwargs.setdefault('dimension', 3)))
        self.noise = kwargs.setdefault('noise', 0.0)
        self.reward_noise = kwargs.setdefault('reward_noise', 0.0)
        self.random_start = kwargs.setdefault('random_start', False)
        self.state = np.zeros((dim-1, 2))
        self.state_range = np.array(
            [[[-1.2, 0.6], [-0.07, 0.07]] for i in range(dim-1)])
        self.goalPos = 0.5
        self.acc = 0.001
        self.gravity = -0.0025
        self.hillFreq = 3.0
        self.delta_time = 1.0

    def reset(self):
        if self.random_start:
            self.state = np.random.random(self.state.shape)
            self.state *= (self.state_range[:, :, 1] -
                           self.state_range[:, :, 0])
            self.state += self.state_range[:, :, 0]
        else:
            self.state = np.zeros(self.state.shape)
            self.state[:, 0] = -0.5

    def isAtGoal(self):
        return (self.state[:, 0] >= self.goalPos).all()

    def takeAction(self, intAction):
        # Translate action into a (possibly) multi-dimensional direction
        intAction = np.array(intAction).flatten()[0]
        direction = np.zeros((self.state.shape[0],))  # Zero is Neutral
        if intAction >= 0:
            direction[int(intAction)/2] = ((intAction % 2) - 0.5)*2.0
        if self.noise > 0:
            direction += self.acc * np.random.normal(
                scale=self.noise, size=direction.shape)

        self.state[:, 1] += \
            self.acc*(direction) + \
            self.gravity*np.cos(self.hillFreq*self.state[:, 0])
        self.state[:, 1] = self.state[:, 1].clip(
            min=self.state_range[:, 1, 0], max=self.state_range[:, 1, 1])
        self.state[:, 0] += self.delta_time * self.state[:, 1]
        self.state[:, 0] = self.state[:, 0].clip(
            min=self.state_range[:, 0, 0], max=self.state_range[:, 0, 1])

    def step(self, action, step_penalty_coeff=1.0, distance_coeff=0.0):
        done = False
        reward = - step_penalty_coeff - distance_coeff * (self.state[:, 0] - self.goalPos) ** 2

        self.takeAction(action)

        if self.isAtGoal():
            reward = 0.0
            done = True

        if self.reward_noise > 0:
            reward += np.random.normal(scale=self.reward_noise)

        next_state = np.copy(self.state).flatten(-1)
        next_obs = np.copy(next_state)
        return next_state, next_obs, reward, done


class MountainCar(MountainCarND):
    name = "Mountain Car"

    def __init__(self, **kwargs):
        kwargs['dimension'] = 2
        super(MountainCar, self).__init__(**kwargs)


class MountainCarMDP(MDP):

    @autoargs.arg('step_penalty_coeff', type=float,
                  help='Coefficient for the step penalty')
    @autoargs.arg('distance_coeff', type=float,
                  help='Coefficient for the penalty for being far away from the goal')
    def __init__(
            self,
            step_penalty_coeff=1,
            distance_coeff=0):
        self._mc = MountainCar(random_start=True)
        self.step_penalty_coeff = step_penalty_coeff
        self.distance_coeff = distance_coeff
        self.reset()

    @overrides
    def reset(self):
        self._mc.reset()
        return self._mc.state.reshape(-1), self._mc.state.reshape(-1)

    @property
    @overrides
    def observation_dtype(self):
        return 'float32'

    @property
    @overrides
    def observation_shape(self):
        return self._mc.state.reshape(-1).shape

    @property
    @overrides
    def action_dim(self):
        return 2

    @property
    @overrides
    def action_dtype(self):
        return 'uint8'

    @overrides
    def step(self, state, action):
        return self._mc.step(
                action,
                step_penalty_coeff=self.step_penalty_coeff,
                distance_coeff=self.distance_coeff)
