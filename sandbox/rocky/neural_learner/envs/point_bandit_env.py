import numpy as np
from cached_property import cached_property

from rllab.envs.base import Env, Step
from rllab.spaces import Box
from sandbox.rocky.neural_learner.envs.mab_env import MABEnv
from sandbox.rocky.tf.envs.vec_env import VecEnv


class PointBanditEnv(Env):
    def __init__(self, n_arms=10, side_length=1, max_action=0.1):
        self.n_arms = n_arms
        self.side_length = side_length
        self.pt_radius = None
        self.pt_centers = None
        self.min_x = None
        self.min_y = None
        self.max_x = None
        self.max_y = None
        self.max_action = max_action
        self.initialize()
        self.executor = VecPointBandit(env=self, n_envs=1)
        self.reset_trial()

    def initialize(self):
        self.min_x = -self.side_length / 2
        self.min_y = -self.side_length / 2
        self.max_x = self.side_length / 2
        self.max_y = self.side_length / 2
        max_radius = self.side_length / 2 * 0.3

        angle_per_arm = 2 * np.pi / self.n_arms
        radius = (self.max_x - self.min_x) / 2 * 0.9

        inter_pt_dist = np.sin(angle_per_arm / 2) * radius * 2
        pt_center_dist = np.cos(angle_per_arm / 2) * radius

        pt_radius = min(min(inter_pt_dist / 2, pt_center_dist) * 0.9, max_radius)

        pt_centers = []

        for idx in range(self.n_arms):
            pt_angle = idx * angle_per_arm
            pt_center = (np.cos(pt_angle) * radius, np.sin(pt_angle) * radius)
            pt_centers.append(pt_center)

        self.pt_radius = pt_radius
        self.pt_centers = np.asarray(pt_centers)

    def reset_trial(self):
        return self.executor.reset_trial([True])[0]

    def reset(self):
        return self.executor.reset([True])[0]

    def render(self, close=False):
        import gizeh
        import cv2

        if close:
            cv2.destroyWindow("image")

        width = 400
        height = 400

        def rescale_size(size):
            return size * width / (self.max_x - self.min_x)

        def rescale_point(x, y):
            tx = (x - self.min_x) / (self.max_x - self.min_x) * height
            ty = (y - self.min_y) / (self.max_y - self.min_y) * width
            return tx, ty

        surface = gizeh.Surface(width=width, height=height, bg_color=(1, 1, 1))

        gizeh.circle(
            r=rescale_size(self.side_length * 0.05),
            xy=rescale_point(*self.executor.agent_poses[0]),
            fill=(0, 0, 0)
        ).draw(surface)

        for pt_center in self.pt_centers:
            gizeh.circle(
                r=rescale_size(self.pt_radius),
                xy=rescale_point(*pt_center),
                fill=(1, 0, 0)
            ).draw(surface)

        img = surface.get_npimage()

        cv2.imshow("image", img)
        cv2.waitKey(10)

    @cached_property
    def observation_space(self):
        return Box(
            low=np.array([self.min_x, self.min_y]),
            high=np.array([self.max_x, self.max_y])
        )

    @cached_property
    def action_space(self):
        return Box(low=-self.max_action, high=self.max_action, shape=(2,))

    def step(self, action):
        next_obs, rewards, dones, infos = self.executor.step([action], max_path_length=None)
        return next_obs[0], rewards[0], dones[0], {k: v[0] for k, v in infos.items()}

    def vec_env_executor(self, n_envs):
        return VecPointBandit(env=self, n_envs=n_envs)

    @property
    def vectorized(self):
        return True


class VecPointBandit(VecEnv):
    def __init__(self, env, n_envs):
        self.env = env
        self.n_envs = n_envs
        self.vec_mab_env = MABEnv(n_arms=self.env.n_arms).vec_env_executor(n_envs=n_envs)
        self.ts = np.zeros((self.n_envs,))
        self.agent_poses = np.zeros((self.n_envs, 2))
        self.reset_trial(dones=[True] * n_envs)

    def reset_trial(self, dones, seeds=None, *args, **kwargs):
        self.vec_mab_env.reset_trial(dones, seeds=seeds, *args, **kwargs)
        return self.reset(dones=dones, seeds=seeds, *args, **kwargs)

    def reset(self, dones, seeds=None, *args, **kwargs):
        dones = np.cast['bool'](dones)
        self.ts[dones] = 0
        self.agent_poses[dones] = 0
        return self.agent_poses[dones]

    def step(self, actions, max_path_length):
        actions = np.clip(
            actions,
            -self.env.max_action,
            self.env.max_action,
        )
        self.agent_poses = np.clip(
            self.agent_poses + actions,
            self.env.observation_space.low,
            self.env.observation_space.high,
        )
        dists = np.linalg.norm(
            self.agent_poses[:, None, :] - self.env.pt_centers[None, :, :],
            axis=-1
        )
        self.ts += 1
        within_bound = dists < self.env.pt_radius
        rewards = np.zeros((self.n_envs,))
        dones = np.cast['bool']([False] * self.n_envs)
        env_ids, arm_ids = np.where(within_bound)
        if len(env_ids) > 0:
            p = self.vec_mab_env.arm_means[env_ids, arm_ids]
            rewards[env_ids] = np.random.binomial(n=1, p=p)
            dones[env_ids] = True
        if max_path_length is not None:
            dones[self.ts >= max_path_length] = True
        if np.any(dones):
            self.reset(dones)
        return np.copy(self.agent_poses), rewards, dones, dict()


if __name__ == "__main__":
    from rllab.sampler.utils import rollout
    from rllab.policies.uniform_control_policy import UniformControlPolicy
    from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler

    env = PointBanditEnv(n_arms=5, side_length=2, max_action=0.1)
    policy = UniformControlPolicy(env.spec)
    while True:
        rollout(env=env, agent=policy, max_path_length=100, animated=True)
