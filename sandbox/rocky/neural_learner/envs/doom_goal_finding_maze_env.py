from cached_property import cached_property
import os
import uuid
import numpy as np
from rllab import config
from rllab.core.serializable import Serializable
from vizdoom import ScreenResolution
from vizdoom import ScreenFormat
from vizdoom import Button
from vizdoom import Mode

from rllab.misc import logger
from sandbox.rocky.neural_learner.doom_utils.wad import WAD
from sandbox.rocky.neural_learner.envs.doom_env import DoomEnv


class DoomGoalFindingMazeEnv(DoomEnv, Serializable):
    def __init__(self, restart_game=True):
        Serializable.quick_init(self, locals())
        DoomEnv.__init__(self, restart_game=restart_game)

    def get_doom_config(self):
        DOOM_PATH = os.environ["DOOM_PATH"]

        doom_config = DoomEnv.get_doom_config(self)

        wad = WAD.from_folder(
            os.path.join(config.PROJECT_PATH, "sandbox/rocky/neural_learner/envs/wads/goal_finding_maze"))
        wad_file_name = "/tmp/%s.wad" % uuid.uuid4()
        wad.save(wad_file_name, force=True)
        doom_config.vizdoom_path = os.path.join(DOOM_PATH, "bin/vizdoom").encode("utf-8")
        doom_config.doom_game_path = os.path.join(DOOM_PATH, "scenarios/freedoom2.wad").encode("utf-8")
        doom_config.doom_scenario_path = wad_file_name.encode("utf-8")
        doom_config.doom_map = b"MAP01"
        doom_config.render_hud = False
        doom_config.render_crosshair = False
        doom_config.render_weapon = False
        doom_config.render_decals = False
        doom_config.render_particles = False
        doom_config.living_reward = -0.01
        doom_config.available_buttons = [
            Button.TURN_LEFT,
            Button.TURN_RIGHT,
            Button.MOVE_FORWARD,
            Button.MOVE_BACKWARD
        ]
        return doom_config

    @cached_property
    def action_map(self):
        return np.asarray([
            [True, False, False, False],
            [False, True, False, False],
            [False, False, True, False],
            [False, False, False, True],
        ], dtype=np.intc)

    def log_diagnostics(self, paths):
        success_rate = np.mean([path["rewards"][-1] > 0 for path in paths])
        logger.record_tabular('SuccessRate', success_rate)
