from .base import MDP
from ale_python_interface import ALEInterface, ALEState
from itertools import imap
import numpy as np
import operator
import random
import sys
from contextlib import contextmanager

OBS_RAM = 0
OBS_IMAGE = 1

def sequence_equal(s1, s2):
    return len(s1) == len(s2) and all(imap(operator.eq, s1, s2))

@contextmanager
def temp_restore_state(ale, state):
    bk = None
    if state:
        bk = ale.cloneState()
        if bk != state:
            ale.restoreState(state)
    yield
    if state:
        if bk != state:
            ale.restoreState(bk)

class AtariMDP(MDP):

    def __init__(self, rom_path, obs_type=OBS_RAM, stop_per_life=True, cutoff_frame=18000, default_frame_skip=4):
        self._rom_path = rom_path
        self._obs_type = obs_type
        ale = self._new_ale()
        self._action_set = [ale.getMinimalActionSet()]
        self._obs_shape = self.to_obs(ale).shape
        self._stop_per_life = stop_per_life
        self._ales = []
        self._states = []
        self._life_counts = []
        self._cutoff_frame = cutoff_frame
        self._default_frame_skip = default_frame_skip

    def _new_ale(self):
        ale = ALEInterface()
        ale.loadROM(self._rom_path)
        return ale

    # s1, s2 should both be a list of ALEState

    def _reset_ales(self, states):
        # only do the reset if we actually need it
        if not sequence_equal(self._states, states):
            self._ales = map(lambda x: x.ale, states)
            for ale, state in zip(self._ales, states):
                ale.restoreState(state)
            self._states = states

    def to_obs(self, ale, state=None):
        if self._obs_type is OBS_IMAGE:
            return self.to_rgb(ale, state)
        elif self._obs_type is OBS_RAM: 
            return self.to_ram(ale, state)
        else:
            return None

    def step_single(self, state, action, repeat=None):
        next_states, obs, rewards, dones, effective_steps = self.step([state], map(lambda x: [x], action), repeat)
        return next_states[0], obs[0], rewards[0], dones[0], effective_steps[0]

    def step(self, states, action_indices, repeat=None):
        # if the current states do not match the given argument, we need to
        # reset ale to these states
        repeat = repeat or self._default_frame_skip
        self._reset_ales(states)
        next_states = []
        obs = []
        rewards = []
        dones = []
        steps = []
        for ale, action_idx in zip(self._ales, action_indices[0]):
            reward = 0
            prev_lives = ale.lives()
            per_steps = 0
            done = False
            for _ in xrange(repeat):
                reward += ale.act(self.action_set[0][action_idx])
                per_steps += 1
                if ale.game_over() or ale.getEpisodeFrameNumber() >= self._cutoff_frame:
                    done = True
                    ale.reset_game()
                    break
                elif self._stop_per_life and ale.lives() != prev_lives:
                    done = True
                    break
            next_state = ale.cloneState()
            next_states.append(next_state)
            obs.append(self.to_obs(ale))
            rewards.append(reward)
            dones.append(done)
            steps.append(per_steps)
        self._states = next_states[:]
        return next_states, obs, rewards, dones, steps

    @property
    def support_repeat(self):
        return True

    # return: (states, observations)
    def sample_initial_states(self, n):
        self._ales = [self._new_ale() for _ in xrange(n)]
        self._states = map(lambda x: x.cloneState(), self._ales)
        obs = map(self.to_obs, self._ales)
        return self._states[:], obs

    @property
    def action_set(self):
        return self._action_set

    @property
    def action_dims(self):
        return [len(self._action_set[0])]

    @property
    def observation_shape(self):
        return self._obs_shape

    def to_rgb(self, ale_or_state, state=None):
        if isinstance(ale_or_state, ALEState):
            ale, state = ale_or_state.ale, ale_or_state
        elif isinstance(ale_or_state, ALEInterface):
            ale, state = ale_or_state, state
        else:
            raise ValueError('Invalid first argument: must be either ALEState or ALEInstance')
        with temp_restore_state(ale, state):
            (screen_width,screen_height) = ale.getScreenDims()
            arr = np.zeros((screen_height, screen_width, 4), dtype=np.uint8)
            ale.getScreenRGB(arr)
            # The returned values are in 32-bit chunks. How to unpack them into
            # 8-bit values depend on the endianness of the system
            if sys.byteorder == 'little': # the layout is BGRA
                arr = arr[:,:,2::-1] # (0, 1, 2) <- (2, 1, 0)
            else: # the layout is ARGB (I actually did not test this.
                  # Need to verify on a big-endian machine)
                arr = arr[:,:,1:]
            #img = arr / 255.0
            return arr#img


    def to_ram(self, ale_or_state, state=None):
        if isinstance(ale_or_state, ALEState):
            ale, state = ale_or_state.ale, ale_or_state
        elif isinstance(ale_or_state, ALEInterface):
            ale, state = ale_or_state, state
        else:
            raise ValueError('Invalid first argument: must be either ALEState or ALEInstance')
        with temp_restore_state(ale, state):
            ram_size = ale.getRAMSize()
            ram = np.zeros((ram_size),dtype=np.uint8)
            ale.getRAM(ram)
            MAX_RAM = 255
            ram = (np.array(ram) * 1.0 / MAX_RAM) * 2 - 1
            return ram
