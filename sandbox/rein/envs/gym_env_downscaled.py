


import gym
import gym.envs
import gym.spaces
from gym.monitoring import monitor
import os
import os.path as osp
from rllab.envs.base import Env, Step
from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
from rllab.spaces.discrete import Discrete
from rllab.misc import logger
import logging
import scipy
import numpy as np
import matplotlib


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return gray


def convert_gym_space(space):
    if isinstance(space, gym.spaces.Box):
        return Box(low=space.low, high=space.high)
    elif isinstance(space, gym.spaces.Discrete):
        return Discrete(n=space.n)
    else:
        raise NotImplementedError


class CappedCubicVideoSchedule(object):
    def __call__(self, count):
        return monitor.capped_cubic_video_schedule(count)


class FixedIntervalVideoSchedule(object):
    def __init__(self, interval):
        self.interval = interval

    def __call__(self, count):
        return count % self.interval == 0


class NoVideoSchedule(object):
    def __call__(self, count):
        return False


class GymEnv(Env, Serializable):
    def __init__(self, env_name, record_video=True, video_schedule=None, log_dir=None, record_log=True):
        if log_dir is None:
            if logger.get_snapshot_dir() is None:
                logger.log(
                    "Warning: skipping Gym environment monitoring since snapshot_dir not configured.")
            else:
                log_dir = os.path.join(logger.get_snapshot_dir(), "gym_log")
        Serializable.quick_init(self, locals())

        env = gym.envs.make(env_name)
        self.env = env
        self.env_id = env.spec.id

        monitor.logger.setLevel(logging.WARNING)

        assert not (not record_log and record_video)

        if log_dir is None or record_log is False:
            self.monitoring = False
        else:
            if not record_video:
                video_schedule = NoVideoSchedule()
            else:
                if video_schedule is None:
                    video_schedule = CappedCubicVideoSchedule()
            # add 'force=True' if want overwrite dirs
            self.env.monitor.start(log_dir, video_schedule, force=True)
            self.monitoring = True

        self._observation_space = convert_gym_space(
            gym.spaces.Box(0., 1., (1, 42, 42)))
        self._action_space = convert_gym_space(env.action_space)
        self._horizon = env.spec.timestep_limit
        self._log_dir = log_dir

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def horizon(self):
        return self._horizon

    def reset(self):
        next_obs = self.env.reset()
        next_obs = scipy.misc.imresize(
            next_obs, (42, 42, 3), interp='bilinear', mode=None)
        # next_obs = matplotlib.colors.rgb_to_hsv(next_obs)[:, :, 2]
        next_obs = rgb2gray(next_obs)
        next_obs = next_obs / 256.
        next_obs = next_obs[np.newaxis, :, :]
        return next_obs

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        next_obs = scipy.misc.imresize(
            next_obs, (42, 42, 3), interp='bilinear', mode=None)
        # next_obs = matplotlib.colors.rgb_to_hsv(next_obs)[:, :, 2]
        next_obs = rgb2gray(next_obs)
        next_obs = next_obs / 256.
        next_obs = next_obs[np.newaxis, :, :]
        return Step(next_obs, reward, done, **info)

    def render(self):
        self.env.render()

    def terminate(self):
        if self.monitoring:
            self.env.monitor.close()
            if self._log_dir is not None:
                print("""
    ***************************

    Training finished! You can upload results to OpenAI Gym by running the following command:

    python scripts/submit_gym.py %s

    ***************************
                """ % self._log_dir)