from .base import MDP
from ale_python_interface import ale_lib, ALEInterface, ALEState
from itertools import imap
import numpy as np
import operator
import sys
from misc.console import type_hint
from misc.overrides import overrides
from core.serializable import Serializable
from contextlib import contextmanager


EYE_256 = np.eye(256)

def sequence_equal(s1, s2):
    return len(s1) == len(s2) and all(imap(np.array_equal, s1, s2))


class AtariMDP(MDP, Serializable):

    def __init__(
            self, rom_path, obs_type='ram', terminate_per_life=False,
            horizon=18000/4, frame_skip=4):
        ale_lib.setLoggerLevelError()
        ale = ALEInterface()
        ale.loadROM(rom_path)

        self._ale = ale
        self._rom_path = rom_path
        self._obs_type = obs_type
        self._action_set = ale.getMinimalActionSet()
        self._observation_shape = self.get_observation().shape
        self._terminate_per_life = terminate_per_life
        self._current_state = None
        self._life_count = None
        self._horizon = horizon
        self._frame_skip = frame_skip

        self.reset()
        Serializable.__init__(self, rom_path, obs_type, terminate_per_life, horizon, frame_skip)


    def get_observation(self):
        if self._obs_type == 'image':
            return self.get_image()
        elif self._obs_type == 'ram':
            return self.get_ram()
        else:
            return None

    @overrides
    def step(self, state, action):
        if self._current_state is not state:
            self._ale.load_serialized(state)
        reward = 0
        prev_lives = self._ale.lives()
        done = False
        for _ in range(self._frame_skip):
            reward += self._ale.act(self._action_set[action])
            if self._ale.game_over() or self._ale.getEpisodeFrameNumber() >= self._horizon:
                done = True
                self._ale.reset_game()
                break
            elif self._terminate_per_life and self._ale.lives() != prev_lives:
                done = True
                break
        next_state = self._ale.get_serialized()
        next_observation = self.get_observation()
        self._current_state = next_state
        return next_state, next_observation, reward, done

    def reset(self):
        self._ale.reset_game()
        self._current_state = self._ale.get_serialized()
        return self._current_state, self.get_observation()

    @property
    @overrides
    def action_dim(self):
        return len(self._action_set)

    @property
    @overrides
    def observation_shape(self):
        return self._observation_shape

    #@classmethod
    #def to_rgb(cls, ale):
    #    (screen_width, screen_height) = ale.getScreenDims()
    #    arr = np.zeros((screen_height, screen_width, 4), dtype=np.uint8)
    #    ale.getScreenRGB(arr)
    #    # The returned values are in 32-bit chunks. How to unpack them into
    #    # 8-bit values depend on the endianness of the system
    #    if sys.byteorder == 'little':  # the layout is BGRA
    #        arr = arr[:, :, 2::-1]  # (0, 1, 2) <- (2, 1, 0)
    #    # the layout is ARGB (I actually did not test this.
    #    # Need to verify on a big-endian machine)
    #    else:
    #        arr = arr[:, :, 1:]
    #    return arr

    def get_ram(self):
        ram_size = self._ale.getRAMSize()
        ram = np.zeros(ram_size, dtype=np.uint8)
        self._ale.getRAM(ram)
        indices = [3*16 + 3, 3*16 + 12, 3*16 + 1, 3*16 + 6, 0+12, 3*16 + 8,3*16 + 2, 1*16 + 5]
        return np.concatenate(map(EYE_256.__getitem__, ram[indices]))
