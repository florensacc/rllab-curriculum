import numpy as np
from .base import MDP
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
from rllab.misc import special


MAPS = {
    "4x4_safe": [
        "SFFF",
        "FWFW",
        "FFFW",
        "WFFG"
    ],
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ],
    "8x8": [
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


class GridWorldMDP(MDP, Serializable):
    """
    S : starting point
    F : free space
    W : wall
    H : hole (terminates episode)
    G : goal


    """
    def __init__(self, desc='4x4'):
        Serializable.quick_init(self, locals())
        if isinstance(desc, basestring):
            desc = MAPS[desc]
        self._desc = np.array(map(list, desc))
        n_row, n_col = self._desc.shape
        self._n_row = n_row
        self._n_col = n_col
        (start_x,), (start_y,) = np.nonzero(self._desc == 'S')
        self._start_state = np.array([start_x, start_y])
        self._start_state.flags.writeable = False
        self._state = None
        self._domain_fig = None

    def reset(self):
        self._state = self._start_state
        return self._get_current_obs()

    def step(self, action):
        """
        action map:
        0: up
        1: right
        2: down
        3: left
        :param action: should be a one-hot vector encoding the action
        :return:
        """
        action_idx = special.from_onehot(action)
        increments = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])
        next_state = np.clip(
            self._state + increments[action_idx],
            [0, 0],
            [self._n_row - 1, self._n_col - 1]
        )
        done = False
        reward = 0
        next_state_type = self._desc[next_state[0], next_state[1]]
        if next_state_type == 'H':
            done = True
        elif next_state_type == 'W':
            # print "Hid wall!"
            next_state = self._state
        elif next_state_type == 'G':
            done = True
            reward = 1
        self._state = next_state
        return self._get_current_obs(), reward, done

    def _get_current_obs(self):
        return special.to_onehot(self._state[0] * self._n_col + self._state[1], self._n_row * self._n_col)

    @property
    @overrides
    def action_dim(self):
        return 4

    @property
    @overrides
    def action_dtype(self):
        return 'uint8'

    @property
    @overrides
    def observation_dtype(self):
        return 'uint8'

    @property
    def observation_shape(self):
        return (self._n_row * self._n_col,)

    def plot(self):
        import matplotlib as mpl
        mpl.rcParams['toolbar'] = 'None'
        import matplotlib.pyplot as plt
        from matplotlib import colors
        from matplotlib import cm

        # Wall: black
        # Hole: red
        # Agent: green

        level_map = 0 * (self._desc == 'F') + 5 * (self._desc == 'W') + 4 * (self._desc == 'H')
        # print level_map
        # print self._state
        if self._domain_fig is None:
            cmap = colors.ListedColormap(
                ['w', '.75', 'b', 'g', 'r', 'k'], 'GridWorld')
            cm.register_cmap(cmap=cmap)
            self._agent_fig = plt.figure("Domain")
            plt.ion()
            plt.show()
            self._domain_fig = plt.imshow(
                level_map,#self._desc == 'F',
                cmap='GridWorld',
                interpolation='nearest',
                vmin=0,
                vmax=5)
            plt.xticks(np.arange(self._n_col), fontsize=15)
            plt.yticks(np.arange(self._n_row), fontsize=15)
            plt.tight_layout()
            self._agent_fig = plt.gca(
            ).plot(self._state[1],#s[1],
                   self._state[0],#s[0],
                   'bd',
                   markersize=20.0 - self._n_col)
            plt.draw()#show(block=False)#idraw()
            plt.pause(0.001)
        else:
            self._domain_fig.set_data(level_map)#self._desc == 'F')
            self._agent_fig.pop(0).remove()
            self._agent_fig = plt.figure("Domain")
            self._agent_fig = plt.gca(
            ).plot(self._state[1],#s[1],
                   self._state[0],#s[0],
                   'bd',
                   markersize=20.0 - self._n_col)
            plt.draw()

        # #mapcopy = copy(self.map)
        # #mapcopy[s[0],s[1]] = self.AGENT
        # # self.domain_fig.set_data(mapcopy)
        # # Instead of '>' you can use 'D', 'o'
        # self.agent_fig = plt.gca(
        # ).plot(s[1],
        #        s[0],
        #        'k>',
        #        markersize=20.0 - self.COLS)
        # plt.draw()
        #
        # print self._state
        # pass
