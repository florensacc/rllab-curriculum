

from rllab.envs.grid_world_env import GridWorldEnv
from rllab.spaces.box import Box
from rllab.envs.base import Step
from rllab.core.serializable import Serializable
import itertools
import numpy as np
import random
from rllab.misc import logger

AGENT = 0
GOAL = 1
WALL = 2
HOLE = 3
N_OBJECT_TYPES = 4


class ImageGridWorld(GridWorldEnv):
    def __init__(self, desc):
        super(ImageGridWorld, self).__init__(desc)
        self._observation_space = Box(low=0., high=1., shape=(self.n_row, self.n_col, N_OBJECT_TYPES))
        self._original_obs_space = GridWorldEnv.observation_space.fget(self)

    @property
    def observation_space(self):
        return self._observation_space

    def reset(self):
        super(ImageGridWorld, self).reset()
        return self.get_current_obs()

    def step(self, action):
        _, reward, done, info = super(ImageGridWorld, self).step(action)
        agent_state = self._original_obs_space.flatten(self.state)
        return Step(self.get_current_obs(), reward, done, **dict(info, agent_state=agent_state))

    def get_current_obs(self):
        ret = np.zeros(self._observation_space.shape)
        ret[self.desc == 'H', HOLE] = 1
        ret[self.desc == 'W', WALL] = 1
        ret[self.desc == 'G', GOAL] = 1
        cur_x = self.state / self.n_col
        cur_y = self.state % self.n_col
        ret[cur_x, cur_y, AGENT] = 1
        return ret


class RandomImageGridWorld(ImageGridWorld, Serializable):
    def __init__(self, base_desc):
        Serializable.quick_init(self, locals())
        base_desc = np.asarray(list(map(list, base_desc)))
        base_desc[base_desc == 'F'] = '.'
        self.base_desc = base_desc
        self.valid_positions = list(zip(*np.where(base_desc == '.')))
        self.reset()

    def reset(self):
        start_pos, end_pos = random.sample(self.valid_positions, k=2)
        desc = np.copy(self.base_desc)
        desc[start_pos] = 'S'
        desc[end_pos] = 'G'
        ImageGridWorld.__init__(self, desc)
        return ImageGridWorld.reset(self)


class CurriculumRandomImageGridWorld(ImageGridWorld, Serializable):
    def __init__(self, base_desc, n_itr, sample_mode='uniform_in_dist', interp_mode='linear'):
        Serializable.quick_init(self, locals())
        base_desc = np.asarray(list(map(list, base_desc)))
        base_desc[base_desc == 'F'] = '.'
        self.base_desc = base_desc


        self.valid_positions = list(zip(*np.where(base_desc == '.')))
        self.valid_pairs = dict(sorted(
            [(x[0], list(x[1])) for x in itertools.groupby(
                    sorted(
                        list(itertools.permutations(self.valid_positions, 2)),
                        key=self.pos_dist
                    ),
                    self.pos_dist
                )],
            key=lambda x: x[0]
        ))

        self.n_itr = n_itr
        self.sample_mode = sample_mode
        self.interp_mode = interp_mode
        self.itr = 0
        self.current_pos_dist = None
        self.reset()

    def set_iteration(self, itr):
        self.itr = itr

    def pos_dist(self, entry):
            (sx, sy), (ex, ey) = entry
            return abs(sx - ex) + abs(sy - ey)

    @property
    def sampling_range(self):
        return min(
            len(self.valid_pairs),
            int(np.ceil((self.itr + 1) * 1.0 / self.n_itr * len(self.valid_pairs)))
        )

    def sample_pair(self):
        assert self.interp_mode == 'linear' and self.sample_mode == 'uniform_in_dist'
        # first determine the range of sampling
        sampling_range = min(
            len(self.valid_pairs),
            int(np.ceil((self.itr + 1) * 1.0 / self.n_itr * len(self.valid_pairs)))
        )
        dist = random.randint(1, sampling_range)
        return random.choice(self.valid_pairs[dist])

    def reset(self):
        start_pos, end_pos = self.sample_pair()
        desc = np.copy(self.base_desc)
        desc[start_pos] = 'S'
        desc[end_pos] = 'G'
        self.current_pos_dist = self.pos_dist((start_pos, end_pos))
        ImageGridWorld.__init__(self, desc)
        return ImageGridWorld.reset(self)

    def step(self, action):
        next_obs, reward, done, info = ImageGridWorld.step(self, action)
        return Step(next_obs, reward, done, pos_dist=self.current_pos_dist)

    def get_env_info(self):
        return dict(pos_dist=self.current_pos_dist)

    def log_diagnostics(self, paths):
        dists = [p['env_infos']['pos_dist'][0] for p in paths]
        logger.record_tabular('EnvItr', self.itr)
        logger.record_tabular('MaxEnvDist', np.max(dists))
        logger.record_tabular('MinEnvDist', np.min(dists))
        logger.record_tabular('AverageEnvDist', np.mean(dists))


if __name__ == "__main__":
    base_map = [
        ".....",
        ".....",
        ".....",
        ".....",
        ".....",
    ]
    env = CurriculumRandomImageGridWorld(base_desc=base_map, n_itr=10)
    env.reset()
