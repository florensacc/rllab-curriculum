import itertools

from rllab.envs.base import Env, Step
import numpy as np
import scipy
from cached_property import cached_property

from rllab.misc import logger
from rllab.spaces.product import Product
from rllab.spaces.box import Box
from sandbox.rocky.analogy.utils import unwrap, using_seed

import math
import numba
import cv2


@numba.njit
def render_image(poses, screen_width, screen_height, colors, buffer):
    image = buffer
    radius = 0.05
    scaled_radius = max(1, int(math.floor(radius * min(screen_width, screen_height))))
    for pos_idx in range(len(poses)):
        x, y = poses[pos_idx]
        color = colors[pos_idx]
        scaled_x = int(np.floor((x + 1) * screen_height * 0.5))
        scaled_y = int(np.floor((y + 1) * screen_width * 0.5))
        cur_radius = scaled_radius
        for x_ in range(max(0, scaled_x - cur_radius), min(screen_height, scaled_x + cur_radius)):
            for y_ in range(max(0, scaled_y - cur_radius), min(screen_width, scaled_y + cur_radius)):
                image[x_, y_] = color
    return image


def rand_unique_seq(choices, seq_length):
    seq = np.random.choice(
        choices,
        replace=True,
        size=seq_length
    )
    while True:
        # make sure no adjacent targets are the same
        repeats = np.asarray([idx
                              for idx, x, y in zip(itertools.count(), seq, seq[1:])
                              if x == y])
        if len(repeats) == 0:
            break
        # regenerate these
        seq[repeats] = np.random.choice(choices, replace=True, size=len(repeats))
    return seq


def make_colors():
    colors = [
        '#1f77b4',  # muted blue
        '#ff7f0e',  # safety orange
        '#2ca02c',  # cooked asparagus green
        '#d62728',  # brick red
        '#9467bd',  # muted purple
        '#8c564b',  # chestnut brown
        '#e377c2',  # raspberry yogurt pink
        '#7f7f7f',  # middle gray
        '#bcbd22',  # curry yellow-green
        '#17becf'  # blue-teal
    ]

    def to_rgb(color):
        hex = int(color[1:], base=16)
        r = hex >> 16
        g = (hex >> 8) & 255
        b = hex & 255
        return r / 255., g / 255., b / 255.

    return list(map(to_rgb, colors))


COLORS = make_colors()


class Shuffler(object):
    def shuffle(self, demo_paths, analogy_paths, demo_seeds, analogy_seeds, target_seeds):
        # We are free to swap the pairs as long as they correspond to the same task
        target_ids_list = [p["env_infos"]["target_ids"][0] for p in analogy_paths]
        target_ids_strs = np.asarray([",".join(map(str, x)) for x in target_ids_list])

        for target_ids in set(target_ids_strs):
            # shuffle each set of tasks separately
            matching_ids, = np.where(target_ids_strs == target_ids)
            shuffled = np.copy(matching_ids)
            np.random.shuffle(shuffled)
            analogy_paths[matching_ids] = analogy_paths[shuffled]
            analogy_seeds[matching_ids] = analogy_seeds[shuffled]


class SimpleParticleEnv(Env):
    def __init__(
            self,
            n_particles=2,
            min_seq_length=1,
            max_seq_length=1,
            seed=None,
            target_seed=None,
            n_vis_demo_segments=100,
            min_margin=None,
            obs_type='full_state',
            obs_size=(100, 100),
            random_init_position=True
    ):
        """
        :param n_particles: Number of particles
        :param seed: Seed for generating positions of the particles
        :param target_seed: Seed for generating the target particle
        :param n_vis_demo_segments: Number of segments to visualize
        :param min_margin: Minimum margin between any pair of particles. Increase this parameter to disambiguate
        between different possible goals
        :param obs_type: either 'state' or 'image'
        :param obs_size: only used when using image observations. Should be a tuple of the form (height, width)
        :return:
        """
        self.seed = seed
        self.particles = None
        self.n_particles = n_particles
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.agent_pos = None
        self.viewers = dict()
        self.target_ids = None
        self.target_id = None
        self.target_index = None
        self.target_seed = target_seed
        self.n_vis_demo_segments = n_vis_demo_segments
        if min_margin is None:
            min_margin = (((0.8 * 2) ** 2 / (n_particles + 1)) ** 0.5) / 2
        self.position_bounds = (-1, 1)
        self.action_bounds = (-0.1, 0.1)
        self.min_margin = min_margin
        self.obs_type = obs_type
        self.obs_size = obs_size
        self.random_init_position = random_init_position
        self._state_obs_space = Product(
            Box(low=-np.inf, high=np.inf, shape=(2,)),
            Box(low=-np.inf, high=np.inf, shape=(self.n_particles, 2))
        )
        self._image_obs_space = Box(low=-1, high=1, shape=self.obs_size + (3,))
        self.reset()

    def reset_trial(self):
        seed = np.random.randint(np.iinfo(np.int32).max)
        self.seed = seed
        target_seed = np.random.randint(np.iinfo(np.int32).max)
        self.target_seed = target_seed
        return self.reset()

    def reset(self):
        seed = self.seed
        with using_seed(seed):
            if self.random_init_position:
                n = self.n_particles + 1
            else:
                n = self.n_particles

            particles = np.random.uniform(
                low=-0.8, high=0.8, size=(n, 2)
            )
            if self.min_margin > 0:
                while True:
                    l2_in_conflict = np.where(
                        scipy.spatial.distance.squareform(
                            scipy.spatial.distance.pdist(particles, 'euclidean')
                        ) + np.eye(n) * 10000 < self.min_margin
                    )
                    if len(l2_in_conflict[0]) > 0:
                        tweak_idx = l2_in_conflict[0][0]
                        particles[tweak_idx] = np.random.uniform(low=-0.8, high=0.8, size=(2,))
                    else:
                        break

            if self.random_init_position:
                self.particles = particles[:-1]
                self.agent_pos = particles[-1]
            else:
                self.agent_pos = np.array([0., 0.])
                self.particles = particles

        with using_seed(self.target_seed):
            seq_length = np.random.randint(low=self.min_seq_length, high=self.max_seq_length + 1)
            self.target_ids = rand_unique_seq(
                np.arange(self.n_particles),
                seq_length
            )

        self.target_id = self.target_ids[0]
        self.target_index = 0

        return self.get_current_obs()

    def step(self, action):
        done = False
        self.agent_pos += np.asarray(np.clip(action, *self.action_bounds))
        dist = np.sqrt(np.sum(np.square(self.agent_pos - self.particles[self.target_id])))
        reward = -dist
        if dist < 0.1:
            self.target_index += 1
            reward = 1.
            if self.target_index < len(self.target_ids):
                self.target_id = self.target_ids[self.target_index]
            else:
                done = True
        self.agent_pos = np.clip(self.agent_pos, *self.position_bounds)
        return Step(self.get_current_obs(), reward, done, **self.get_env_info())

    @cached_property
    def observation_space(self):
        if self.obs_type == 'full_state':
            return Product(
                Box(low=-np.inf, high=np.inf, shape=(2,)),
                Box(low=-np.inf, high=np.inf, shape=(self.n_particles, 2))
            )
        elif self.obs_type == 'image':
            return Box(low=-1, high=1, shape=self.obs_size + (3,))
        else:
            raise NotImplementedError

    def get_env_info(self):
        return dict(
            agent_pos=np.copy(self.agent_pos),
            target_ids=self.target_ids
        )

    @cached_property
    def action_space(self):
        return Box(low=self.action_bounds[0], high=self.action_bounds[1], shape=(2,))

    def get_state_obs(self):
        return np.copy(self.agent_pos), np.copy(self.particles)

    def get_image_obs(self, rescale=False):
        colors = np.cast['float32'](np.concatenate([np.asarray(COLORS[:self.n_particles]) * 255, np.array([[0, 0, 0]])],
                                                   axis=0))
        poses = np.concatenate([self.particles, [self.agent_pos]], axis=0)

        buffer = np.zeros(self.obs_size + (3,), dtype=np.float32) + 255
        screen_height, screen_width = self.obs_size

        image = render_image(poses=poses, screen_width=screen_width, screen_height=screen_height, colors=colors,
                             buffer=buffer)

        if rescale:
            return ((image / 255.0) - 0.5) * 2
        else:
            return np.cast['uint8'](image)

    def get_current_obs(self):
        if self.obs_type == 'full_state':
            return self.get_state_obs()
        elif self.obs_type == 'image':
            return self.get_image_obs(rescale=True)
        else:
            raise NotImplementedError

    def render(self, mode='human', close=False):
        assert mode == 'human'
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', width=400, height=400)
        cv2.imshow('image', cv2.resize(self.get_image_obs(rescale=False), (400, 400)))
        cv2.waitKey(10)

    def success_rate(self, paths, envs):
        n_visited = [np.sum(np.equal(p["rewards"], 1)) for p in paths]
        seq_lengths = [len(p["env_infos"]["target_ids"][0]) for p in paths]
        success = np.equal(n_visited, seq_lengths)
        return np.mean(success)

    def log_analogy_diagnostics(self, paths, envs):
        n_visited = [np.sum(np.equal(p["rewards"], 1)) for p in paths]
        logger.record_tabular_misc_stat('NVisited', n_visited, placement='front')
        logger.record_tabular('SuccessRate', self.success_rate(paths, envs))

    @classmethod
    def shuffler(cls):
        return Shuffler()


if __name__ == "__main__":

    env = SimpleParticleEnv(n_particles=4, min_seq_length=3, max_seq_length=3)
    env.reset()

    while True:
        import time

        time.sleep(1)
        env.reset_trial()
        env.render()
