from cached_property import cached_property
import os
import numpy as np
from rllab import config
from rllab.core.serializable import Serializable
from sandbox.rocky.neural_learner.envs.doom_env import DoomEnv


class DoomDefaultWadEnv(DoomEnv, Serializable):
    def __init__(self, wad_name="symphod.wad", *args, **kwargs):
        Serializable.quick_init(self, locals())
        self.wad_name = wad_name
        DoomEnv.__init__(self, *args, **kwargs)

    def get_doom_config(self):
        DOOM_PATH = os.environ["DOOM_PATH"]

        doom_config = DoomEnv.get_doom_config(self)

        wad_file_name = os.path.join(config.PROJECT_PATH, "sandbox/rocky/neural_learner/envs/wads/", self.wad_name)
        doom_config.vizdoom_path = os.path.join(DOOM_PATH, "bin/vizdoom").encode("utf-8")
        doom_config.doom_game_path = os.path.join(DOOM_PATH, "scenarios/freedoom2.wad").encode("utf-8")
        doom_config.doom_scenario_path = wad_file_name.encode("utf-8")
        return doom_config

    @cached_property
    def action_map(self):
        return np.asarray([
            [True, False, False, False, False, False],
            [False, True, False, False, False, False],
            [False, False, True, False, False, False],
            [False, False, False, True, False, False],
            [False, False, False, False, True, False],
            [False, False, False, False, False, True],
        ], dtype=np.intc)
