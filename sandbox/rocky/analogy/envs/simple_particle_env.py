from __future__ import print_function
from __future__ import absolute_import
from rllab.envs.base import Env, Step
import random
import numpy as np
import contextlib

from rllab.misc import logger
from rllab.spaces.product import Product
from rllab.spaces.box import Box


@contextlib.contextmanager
def using_seed(seed):
    rand_state = random.getstate()
    np_rand_state = np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    yield
    random.setstate(rand_state)
    np.random.set_state(np_rand_state)


class SimpleParticleEnv(Env):
    # The agent always starts at (0, 0)
    def __init__(self, n_particles=2, seed=None, target_seed=None, n_vis_demo_segments=100):
        self.seed = seed
        self.particles = None
        self.n_particles = n_particles
        self.agent_pos = None
        self.viewer = None
        self.target_id = None
        self.target_seed = target_seed
        self.n_vis_demo_segments = n_vis_demo_segments

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
            self.particles = np.random.uniform(
                low=-0.8, high=0.8, size=(self.n_particles, 2)
            )
            with using_seed(self.target_seed):
                self.target_id = np.random.choice(np.arange(self.n_particles))
        self.agent_pos = np.array([0., 0.])
        return self.get_current_obs()

    def step(self, action):
        self.agent_pos += np.asarray(action)
        dist = np.sqrt(np.sum(np.square(self.agent_pos - self.particles[self.target_id])))
        reward = -dist
        return Step(self.get_current_obs(), reward, False)

    @property
    def observation_space(self):
        return Product(
            Box(low=-np.inf, high=np.inf, shape=(2,)),
            Box(low=-np.inf, high=np.inf, shape=(self.n_particles, 2))
        )

    @property
    def action_space(self):
        return Box(low=-0.1, high=0.1, shape=(2,))

    def get_current_obs(self):
        return np.copy(self.agent_pos), np.copy(self.particles)

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

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

        colors = list(map(to_rgb, colors))

        assert len(colors) >= self.n_particles

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)  # , display=self.display)
            self.target_geoms = []
            self.target_attrs = []
            self.vis_demo_segment_geoms = []
            self.vis_demo_segment_attrs = []
            a = 20
            for idx in range(self.n_particles):
                target_attr = rendering.Transform()
                target_geom = rendering.FilledPolygon([(-a, -a), (-a, a), (a, a), (a, -a)])
                target_geom.add_attr(target_attr)
                target_geom.set_color(*colors[idx])
                self.viewer.add_geom(target_geom)
                self.target_attrs.append(target_attr)
                self.target_geoms.append(target_geom)

            for idx in range(self.n_vis_demo_segments):
                seg_geom = rendering.make_circle(20, res=100)
                seg_attr = rendering.Transform()
                seg_geom._color.vec4 = (0, 0, 0, 0.01)
                seg_geom.add_attr(seg_attr)
                self.vis_demo_segment_geoms.append(seg_geom)
                self.vis_demo_segment_attrs.append(seg_attr)
                self.viewer.add_geom(seg_geom)

            self.agent_geom = rendering.make_circle(20, res=100)
            self.agent_attr = rendering.Transform()
            self.agent_geom.set_color(0, 0, 0)
            self.agent_geom.add_attr(self.agent_attr)
            self.viewer.add_geom(self.agent_geom)

        for pos, target_attr in zip(self.particles, self.target_attrs):
            target_attr.set_translation(
                screen_width / 2 * (1 + pos[0]),
                screen_height / 2 * (1 + pos[1]),
            )
        for idx in range(self.n_vis_demo_segments):
            seg_pos = self.agent_pos + 1.0 * idx / self.n_vis_demo_segments * (self.particles[self.target_id] - self.agent_pos)
            self.vis_demo_segment_attrs[idx].set_translation(
                screen_width / 2 * (1 + seg_pos[0]),
                screen_height / 2 * (1 + seg_pos[1]),
            )

        self.agent_attr.set_translation(
            screen_width / 2 * (1 + self.agent_pos[0]),
            screen_height / 2 * (1 + self.agent_pos[1]),
        )
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def log_analogy_diagnostics(self, paths, envs):
        last_agent_pos = np.asarray([self.observation_space.unflatten(p["observations"][-1])[0] for p in paths])
        target_pos = np.asarray([e.particles[e.target_id] for e in envs])
        dists = np.sqrt(np.sum(np.square(last_agent_pos - target_pos), axis=-1))
        logger.record_tabular('AverageFinalDistToGoal', np.mean(dists))
        logger.record_tabular('SuccessRate(Dist<0.1)', np.mean(dists < 0.1))
        logger.record_tabular('SuccessRate(Dist<0.05)', np.mean(dists < 0.05))
        logger.record_tabular('SuccessRate(Dist<0.01)', np.mean(dists < 0.01))


if __name__ == "__main__":
    env = SimpleParticleEnv()
    env.reset()
    while True:
        import time

        time.sleep(0.01)
        env.render()
        print(env.step(np.random.uniform(low=-0.01, high=0.01, size=(2,))))
