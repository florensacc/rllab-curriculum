from __future__ import print_function
from __future__ import absolute_import
from rllab.envs.base import Env, Step
import random
import numpy as np
import contextlib
import scipy
import math
from cached_property import cached_property
import cv2

from rllab.misc import logger
from rllab.spaces.product import Product
from rllab.spaces.box import Box
from sandbox.rocky.analogy.utils import unwrap
from sandbox.rocky.tf.misc import tensor_utils
import tensorflow as tf
from tensorflow.python.ops import gen_state_ops


def decompress_naive(observations, screen_width, screen_height, colors, buffer):
    images = buffer
    radius = 0.05
    # scaled_diameter = int(math.ceil((radius * 2) * min(screen_width, screen_height)))
    scaled_radius = max(1, int(math.floor(radius * min(screen_width, screen_height))))
    for idx, obs in enumerate(observations):
        poses = obs.reshape((-1, 2))
        image = images[idx]
        for pos, color in zip(poses, colors):
            scaled_center = (pos + 1) * np.array([screen_height, screen_width]) * 0.5
            scaled_center = np.cast['int'](np.floor(scaled_center))
            image[
            scaled_center[0] - scaled_radius:scaled_center[0] + scaled_radius,
            scaled_center[1] - scaled_radius:scaled_center[1] + scaled_radius,
            ] = color
    return images


import numba


@numba.njit
def decompress_numba1(observations, screen_width, screen_height, colors, buffer):
    images = buffer
    radius = 0.05
    # scaled_diameter = int(math.ceil((radius * 2) * min(screen_width, screen_height)))
    scaled_radius = max(1, int(math.floor(radius * min(screen_width, screen_height))))
    for idx in range(len(observations)):
        obs = observations[idx]
        poses = obs.reshape((-1, 2))
        image = images[idx]
        for pos_idx in range(len(poses)):
            x, y = poses[pos_idx]
            color = colors[pos_idx]
            scaled_x = int(np.floor((x + 1) * screen_height * 0.5))
            scaled_y = int(np.floor((y + 1) * screen_width * 0.5))
            for x_ in range(scaled_x - scaled_radius, scaled_x + scaled_radius):
                for y_ in range(scaled_y - scaled_radius, scaled_y + scaled_radius):
                    image[x_, y_] = color
    return images


@contextlib.contextmanager
def using_seed(seed):
    rand_state = random.getstate()
    np_rand_state = np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    yield
    random.setstate(rand_state)
    np.random.set_state(np_rand_state)


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
    def shuffle(self, demo_paths, analogy_paths, demo_envs, analogy_envs):
        # We are free to swap the pairs as long as they correspond to the same task
        target_ids = [unwrap(x).target_id for x in analogy_envs]
        for target_id in set(target_ids):
            # shuffle each set of tasks separately
            matching_ids, = np.where(target_ids == target_id)
            shuffled = np.copy(matching_ids)
            np.random.shuffle(shuffled)
            analogy_paths[matching_ids] = analogy_paths[shuffled]
            analogy_envs[matching_ids] = analogy_envs[shuffled]


class SimpleParticleEnv(Env):
    # The agent always starts at (0, 0)
    def __init__(self, n_particles=2, seed=None, target_seed=None, n_vis_demo_segments=100, min_margin=0.,
                 min_angular_margin=0., obs_type='state', obs_size=(100, 100), random_init_position=False):
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
        self.agent_pos = None
        self.viewers = dict()
        self.target_id = None
        self.target_seed = target_seed
        self.n_vis_demo_segments = n_vis_demo_segments
        self.min_margin = min_margin
        self.min_angular_margin = min_angular_margin
        self.obs_type = obs_type
        self.obs_size = obs_size
        self.random_init_position = random_init_position
        self._compressed = False

    @property
    def compressed(self):
        return self._compressed

    @compressed.setter
    def compressed(self, val):
        self._compressed = val

    def decompress(self, observations):
        if self.obs_type == 'state':
            return observations
        if len(observations.shape) == 3:
            flat_obs = observations.reshape((-1, observations.shape[-1]))
        else:
            flat_obs = observations

        # import time
        # @contextlib.contextmanager
        # def time_func(msg):
        #     start = time.time()
        #     yield
        #     end = time.time()
        #     print("%s took %fs" % (msg, end - start))

        # buffer = np.zeros((flat_obs.shape[0],) + self.obs_size + (3,), dtype=np.float32) + 255

        colors = np.cast['float32'](np.concatenate([np.array([[0, 0, 0]]), np.asarray(COLORS) * 255], axis=0))

        # result_2 = decompress_numba1(flat_obs, self.obs_size[0], self.obs_size[1], colors, buffer)
        buffer = np.zeros((flat_obs.shape[0],) + self.obs_size + (3,), dtype=np.float32) + 255
        # with time_func("numba1"):
        #     result_2 = decompress_numba1(flat_obs, self.obs_size[0], self.obs_size[1], colors, buffer)
        decompressed = decompress_numba1(flat_obs, self.obs_size[0], self.obs_size[1], colors, buffer)

        if len(observations.shape) == 3:
            return decompressed.reshape(observations.shape[:2] + (-1,))
        else:
            return decompressed.reshape((observations.shape[0], -1))

        # cv2.imshow('image', result_2[0] / 255)
        # cv2.waitKey()
        # import ipdb;
        # ipdb.set_trace()
        # N = flat_obs.shape[0]
        #
        #
        #
        # agent_poses = flat_obs[:, :2]
        # particle_poses = flat_obs[:, 2:].reshape((-1, self.n_particles, 2))
        #
        # buffer = self.build_draw_op()(agent_poses, particle_poses)
        # create batched images from these
        # initialize to a white canvas
        # buffer = np.zeros((N,) + self.obs_size + (3,), dtype=np.uint8) - 1
        #
        #
        #
        # agent_pos = flat_obs[:, :2]
        #
        # buffer = self.batch_draw_rect(buffer, center=agent_pos, radius=0.05, color=np.cast['int']((0, 0, 0)))
        # for idx in range(self.n_particles):
        #     color = np.cast['int'](np.asarray(COLORS[idx]) * 255)
        #     buffer = self.batch_draw_rect(buffer, center=flat_obs[:, (idx + 1) * 2:(idx + 2) * 2], radius=0.05,
        #                          color=color)

        # cv2.imshow('image', buffer[0] / 255)
        # cv2.waitKey()

    # def build_draw_op(self):
    #     # we need the following vars
    #     agent_poses = tf.placeholder(tf.float32, shape=(None, 2), name="agent_pos")
    #     particle_poses = tf.placeholder(tf.float32, shape=(None, self.n_particles, 2), name="particle_poses")
    #     # colors = tf.placeholder(tf.float32, shape=(self.n_particles))
    #     radius = 0.05  # tf.placeholder(tf.float32, shape=(), name="radius")
    #     colors = COLORS  # tf.placeholder(tf.float32, shape=(self.n_particles+1, 3), name="color")
    #     N = tf.shape(agent_poses)[0]
    #     buffer = tf.ones(tf.pack((N,) + self.obs_size + (3,)), dtype=tf.float32) * 255
    #     buffer = self.build_draw_rect_op(buffer, agent_poses, radius, np.array([0, 0, 0]))  # colors[0])
    #     for idx in range(self.n_particles):
    #         buffer = self.build_draw_rect_op(buffer, particle_poses[:, idx, :], radius, colors[idx])
    #     return tensor_utils.compile_function(
    #         inputs=[agent_poses, particle_poses],
    #         outputs=buffer,
    #     )
    #
    # def build_draw_rect_op(self, buffer, centers, radius, color):
    #     N = tf.shape(centers)[0]
    #     screen_height, screen_width = self.obs_size
    #     scaled_diameter = int(np.ceil((radius * 2) * min(screen_width, screen_height)))
    #     scaled_center = (centers + 1) * np.asarray(self.obs_size) * 0.5
    #     scaled_center = tf.cast(tf.floor(scaled_center), tf.int32)
    #     ids_0 = tf.range(N)
    #     ids_1 = scaled_center[:, 0]
    #     ids_2 = scaled_center[:, 1]
    #     flat_ids = ids_0 * screen_height * screen_width + ids_1 * screen_width + ids_2
    #     flat_buffer = tf.reshape(buffer, (-1, len(color)))
    #
    #     import ipdb;
    #     ipdb.set_trace()
    #
    #     diff_canvas = gen_state_ops._temporary_variable(tf.shape(flat_buffer), tf.float32)
    #     diff_canvas = tf.assign(diff_canvas, tf.zeros_like(flat_buffer))
    #     # ids = tf.transpose(tf.pack([ids_0, ids_1, ids_2]))
    #     # fl
    #     # color[None, :] - tf.gather_nd(buffer, ids)
    #
    #     # diff_canvas = tf.zeros_like(flat_buffer)
    #     import ipdb;
    #     ipdb.set_trace()
    #     diff_canvas = tf.scatter_update(diff_canvas, flat_ids, tf.convert_to_tensor(color)[None, :])
    #     diff_canvas = tf.scatter_sub(diff_canvas, flat_ids, flat_buffer)
    #     diff_canvas = tf.reshape(diff_canvas, tf.shape(buffer))
    #     # finally, ready to apply convolution!
    #     filter = np.zeros((scaled_diameter, scaled_diameter, 3, 3))
    #     filter[:, :] = np.eye(3)
    #     diff_canvas = tf.nn.conv2d(diff_canvas, filter, strides=(1, 1, 1, 1), padding='SAME')
    #     return buffer + diff_canvas
    #
    #     # diff_canvas = tf.zeros_like(buffer, dtype=tf.float32)
    #     # diff_canvas = tf.scatter_update(diff_canvas, indices=flat_, )
    #     # buffer
    #     # return None
    #
    # def batch_draw_rect(self, buffer, center, radius, color):
    #     N = buffer.shape[0]
    #     screen_height, screen_width = self.obs_size
    #     scaled_diameter = int(np.ceil((radius * 2) * min(screen_width, screen_height)))
    #     scaled_center = (center + 1) * np.asarray(self.obs_size) * 0.5
    #     # scaled_radius = np.cast['int'](np.ceil(scaled_radius))
    #     scaled_center = np.cast['int'](np.floor(scaled_center))
    #
    #     diff_canvas = np.zeros_like(buffer, dtype=np.int)
    #     diff_canvas[np.arange(N), scaled_center[:, 0], scaled_center[:, 1]] = \
    #         color[None, :] - buffer[np.arange(N), scaled_center[:, 0], scaled_center[:, 1]]
    #
    #     filter = np.zeros((scaled_diameter, scaled_diameter, 3, 3))
    #     filter[:, :] = np.eye(3)
    #
    #     logger.log("launching tf session")
    #
    #     with tf.Session() as sess:
    #         diff_canvas_var = tf.Variable(diff_canvas, dtype=tf.float32)
    #         filter_var = tf.Variable(filter, dtype=tf.float32)
    #         sess.run(tf.initialize_variables([diff_canvas_var, filter_var]))
    #         convolved = sess.run(tf.nn.conv2d(diff_canvas_var, filter_var, strides=(1, 1, 1, 1), padding='SAME'))
    #         return buffer + convolved
    #         #     import ipdb; ipdb.set_trace()
    #         #     pass
    #         #
    #         # # xvalid_onehot = np.zeros((N, screen_height), dtype=np.uint8)
    #         #
    #         # # xvalid_onehot[:, scaled_center] = 1
    #         # # xvalid_onehot[np.arange(N), scaled_center[:, 0]] = 1
    #         #
    #         # import ipdb; ipdb.set_trace()

    def reset_trial(self):
        seed = np.random.randint(np.iinfo(np.int32).max)
        self.seed = seed
        target_seed = np.random.randint(np.iinfo(np.int32).max)
        self.target_seed = target_seed
        return self.reset()

    def reset(self, seed=None):
        if seed is None:
            seed = self.seed
        with using_seed(seed):
            if self.random_init_position:
                self.agent_pos = np.random.uniform(low=-0.4, high=0.4, size=(2,))  # np.array([0., 0.])
            else:
                self.agent_pos = np.array([0., 0.])

            self.particles = np.random.uniform(
                low=-0.8, high=0.8, size=(self.n_particles, 2)
            )
            if self.min_margin > 0 or self.min_angular_margin > 0:
                while True:
                    l2_in_conflict = np.where(
                        scipy.spatial.distance.squareform(
                            scipy.spatial.distance.pdist(self.particles, 'euclidean')
                        ) + np.eye(self.n_particles) * 10000 < self.min_margin
                    )
                    cosine_in_conflict = np.where(
                        scipy.spatial.distance.squareform(
                            scipy.spatial.distance.pdist(self.particles - self.agent_pos.reshape((1, -1)), 'cosine')
                        ) + np.eye(self.n_particles) * 10000 < 1 - math.cos(self.min_angular_margin)
                    )
                    if len(l2_in_conflict[0]) > 0:
                        tweak_idx = l2_in_conflict[0][0]
                        self.particles[tweak_idx] = np.random.uniform(low=-0.8, high=0.8, size=(2,))
                    elif len(cosine_in_conflict[0]) > 0:
                        tweak_idx = cosine_in_conflict[0][0]
                        self.particles[tweak_idx] = np.random.uniform(low=-0.8, high=0.8, size=(2,))
                    else:
                        # check
                        break
                        # pairwist_dist =
            with using_seed(self.target_seed):
                self.target_id = np.random.choice(np.arange(self.n_particles))
        return self.get_current_obs()

    def step(self, action):
        self.agent_pos += np.asarray(action)
        dist = np.sqrt(np.sum(np.square(self.agent_pos - self.particles[self.target_id])))
        reward = -dist
        return Step(self.get_current_obs(), reward, False, **self.get_env_info())

    @property
    def observation_space(self):
        if self.obs_type == 'state' or self.obs_type == 'image' and self.compressed:
            return Product(
                Box(low=-np.inf, high=np.inf, shape=(2,)),
                Box(low=-np.inf, high=np.inf, shape=(self.n_particles, 2))
            )
        elif self.obs_type == 'image':
            return Box(low=-1, high=1, shape=self.obs_size + (3,))
        else:
            raise NotImplementedError

    def get_env_info(self):
        return dict(agent_pos=np.copy(self.agent_pos), target_pos=np.copy(self.particles[self.target_id]))

    @cached_property
    def action_space(self):
        return Box(low=-0.1, high=0.1, shape=(2,))

    def get_current_obs(self):
        if self.obs_type == 'state' or self.obs_type == 'image' and self.compressed:
            return np.copy(self.agent_pos), np.copy(self.particles)
        elif self.obs_type == 'image':
            img = self.render(mode='rgb_array')
            # print((img.flatten() * np.arange(img.size)).sum())
            # cv2.imshow('image',img)
            # import time
            # time.sleep(0.1)
            # # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # # rescale to lie in [-1, 1]
            # # cv2.imshow('image', img)
            # # import ipdb; ipdb.set_trace()
            return (img / 255.0 - 0.5) * 2
        else:
            raise NotImplementedError

    def render(self, mode='human', close=False):
        if close:
            if mode in self.viewers:
                self.viewers[mode].close()
                del self.viewers[mode]
            return

        if mode == 'human':
            screen_width = 600
            screen_height = 400
        else:
            screen_width = self.obs_size[1]
            screen_height = self.obs_size[0]

        assert len(COLORS) >= self.n_particles

        if mode not in self.viewers:
            if mode == 'rgb_array':
                from . import cv2_rendering as rendering
            elif mode == 'human':
                from . import rendering
            else:
                raise NotImplementedError
            self.viewers[mode] = rendering.Viewer(screen_width, screen_height, mode=mode)
            viewer = self.viewers[mode]
            self.target_geoms = []
            self.target_attrs = []
            self.vis_demo_segment_geoms = []
            self.vis_demo_segment_attrs = []
            a = screen_width * 0.05
            for idx in range(self.n_particles):
                target_attr = rendering.Transform()
                target_geom = rendering.FilledPolygon([(-a, -a), (-a, a), (a, a), (a, -a)])
                target_geom.add_attr(target_attr)
                target_geom.set_color(*COLORS[idx])
                viewer.add_geom(target_geom)
                self.target_attrs.append(target_attr)
                self.target_geoms.append(target_geom)

            if mode == 'human':
                for idx in range(self.n_vis_demo_segments):
                    seg_geom = rendering.make_circle(screen_width * 0.05, res=100)
                    seg_attr = rendering.Transform()
                    seg_geom._color.vec4 = (0, 0, 0, 0.01)
                    seg_geom.add_attr(seg_attr)
                    self.vis_demo_segment_geoms.append(seg_geom)
                    self.vis_demo_segment_attrs.append(seg_attr)
                    viewer.add_geom(seg_geom)

            self.agent_geom = rendering.make_circle(screen_width * 0.05, res=100)
            self.agent_attr = rendering.Transform()
            self.agent_geom.set_color(0, 0, 0)
            self.agent_geom.add_attr(self.agent_attr)
            viewer.add_geom(self.agent_geom)

        for pos, target_attr in zip(self.particles, self.target_attrs):
            target_attr.set_translation(
                screen_width / 2 * (1 + pos[0]),
                screen_height / 2 * (1 + pos[1]),
            )

        if mode == 'human':
            for idx in range(self.n_vis_demo_segments):
                seg_pos = self.agent_pos + 1.0 * idx / self.n_vis_demo_segments * (
                    self.particles[self.target_id] - self.agent_pos)
                self.vis_demo_segment_attrs[idx].set_translation(
                    screen_width / 2 * (1 + seg_pos[0]),
                    screen_height / 2 * (1 + seg_pos[1]),
                )

        self.agent_attr.set_translation(
            screen_width / 2 * (1 + self.agent_pos[0]),
            screen_height / 2 * (1 + self.agent_pos[1]),
        )
        return self.viewers[mode].render(return_rgb_array=mode == 'rgb_array')

    def log_analogy_diagnostics(self, paths, envs):
        # import ipdb; ipdb.set_trace()
        # last_agent_pos = np.asarray([self.observation_space.unflatten(p["observations"][-1])[0] for p in paths])
        last_agent_pos = np.asarray([p["env_infos"]["agent_pos"][-1] for p in paths])
        target_pos = np.asarray([p["env_infos"]["target_pos"][-1] for p in paths])
        # target_pos = np.asarray([e.particles[e.target_id] for e in envs])
        dists = np.sqrt(np.sum(np.square(last_agent_pos - target_pos), axis=-1))
        logger.record_tabular('AverageFinalDistToGoal', np.mean(dists))
        logger.record_tabular('SuccessRate(Dist<0.1)', np.mean(dists < 0.1))
        logger.record_tabular('SuccessRate(Dist<0.05)', np.mean(dists < 0.05))
        logger.record_tabular('SuccessRate(Dist<0.01)', np.mean(dists < 0.01))

    @classmethod
    def shuffler(cls):
        return Shuffler()


if __name__ == "__main__":
    import math

    env = SimpleParticleEnv(n_particles=6)  # , min_margin=(2.56 / 6) ** 0.5 / 2, min_angular_margin=math.pi / 6)
    env.reset()
    while True:
        import time

        time.sleep(1)
        env.reset_trial()
        env.render()
        # print(env.step(np.random.uniform(low=-0.01, high=0.01, size=(2,))))
