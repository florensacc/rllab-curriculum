import logging
log_level = logging.getLogger().level
import gym
logging.getLogger().setLevel(log_level)
import tensorflow as tf
import os
os.environ["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu/mesa/"
from mujoco_py import glfw
from rllab.envs.base import Env, Step
from rllab.spaces.box import Box
from rllab.core.serializable import Serializable
from conopt.experiments.A4_particle_analogy import Experiment
import numpy as np
from sandbox.rocky.analogy.utils import unwrap, using_seed
from rllab.envs.gym_env import convert_gym_space
from conopt import cost
from cached_property import cached_property
from conopt.worldgen.objs import Obj


def fast_residual2cost(r, metric):
    if len(r.shape) == 1:
        r = np.expand_dims(r, 0)
    if metric == "L2":
        return 0.5 * np.sum(np.square(r))
    else:
        import ipdb;
        ipdb.set_trace()


def fast_compute_cost(reward_fn, s):
    if isinstance(reward_fn, cost.MulCost):
        return reward_fn.b * fast_compute_cost(reward_fn.a_cost, s)
    elif isinstance(reward_fn, cost.AddCost):
        return fast_compute_cost(reward_fn.a_cost, s) + fast_compute_cost(reward_fn.b_cost, s)
    elif isinstance(reward_fn, cost.DistCost):
        return fast_residual2cost(s[reward_fn.a] - s[reward_fn.b], reward_fn.metric)
    elif isinstance(reward_fn, cost.PenaltyCost):
        return fast_residual2cost(s[reward_fn.element], reward_fn.metric)
    else:
        import ipdb;
        ipdb.set_trace()


class Shuffler(object):
    def shuffle(self, demo_paths, analogy_paths, demo_seeds, analogy_seeds, target_seeds):
        # We are free to swap the pairs as long as they correspond to the same task
        target_ids = [p["env_infos"]["target_id"][0] for p in analogy_paths]
        # unwrap(x).conopt_scenario.task_id for x in
        #               analogy_envs]
        for target_id in set(target_ids):
            # shuffle each set of tasks separately
            matching_ids, = np.where(target_ids == target_id)
            shuffled = np.copy(matching_ids)
            np.random.shuffle(shuffled)
            analogy_paths[matching_ids] = analogy_paths[shuffled]
            analogy_seeds[matching_ids] = analogy_seeds[shuffled]


class ConoptParticleEnv(Env, Serializable):
    def __init__(
            self,
            seed=None,
            target_seed=None,
            obs_type='full_state',
            obs_size=(300, 300),
    ):
        Serializable.quick_init(self, locals())
        self.seed = seed
        self.target_seed = target_seed
        self.conopt_exp = None
        self.conopt_scenario = None
        self.conopt_env = None
        self.obs_type = obs_type
        self.obs_size = obs_size
        import os
        os.environ["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu/mesa/"
        self.reset_trial(seed=seed, target_seed=target_seed)

    def reset_trial(self, seed=None, target_seed=None):
        if seed is None:
            seed = np.random.randint(np.iinfo(np.int32).max)
        if target_seed is None:
            target_seed = np.random.randint(np.iinfo(np.int32).max)
        exp = Experiment()
        with using_seed(target_seed):
            target_id = np.random.randint(0, 2)
        with using_seed(seed):
            if self.obs_type == 'full_state':
                scenario = exp.make_scenario(
                    trial_index=seed,
                    task_id=target_id,
                    obs_type=self.obs_type
                )
            elif self.obs_type == 'image':
                scenario = exp.make_scenario(
                    trial_index=seed,
                    task_id=target_id,
                    obs_type=(self.obs_type,) + self.obs_size,
                )
            else:
                raise NotImplementedError
        env = scenario.to_env()
        self.target_id = target_id
        self.conopt_exp = exp
        self.conopt_scenario = scenario
        self.conopt_env = env
        return self.reset()

    def log_analogy_diagnostics(self, paths, envs):
        pass

    def reset(self):
        return self.convert_obs(self.conopt_env.reset())

    def convert_obs(self, obs):
        if self.obs_type == "full_state":
            return obs
        elif self.obs_type == "image":
            return obs#[2]
        else:
            raise NotImplementedError

    @cached_property
    def observation_space(self):
        if self.obs_type == "full_state":
            return convert_gym_space(self.conopt_env.observation_space)
        elif self.obs_type == "image":
            return convert_gym_space(self.conopt_env.observation_space)#.spaces[2])
        else:
            raise NotImplementedError

    @property
    def action_space(self):
        bounds = self.model.actuator_ctrlrange
        lb = bounds[:, 0]
        ub = bounds[:, 1]
        return Box(lb, ub)

    def render(self, mode='human', close=False):#*args, **kwargs):
        env = self.conopt_env
        if close:
            if 'viewer' in env.__dict__:
                env.viewer.close()
                del env.viewer
        else:
            img = env.world.model.render(np.expand_dims(env.x, 0))[0]
            if mode == 'human':
                import cv2
                img = cv2.resize(img, (300, 300))
                if not 'viewer' in env.__dict__:
                    from gym.envs.classic_control.rendering import SimpleImageViewer
                    env.viewer = SimpleImageViewer()
                env.viewer.imshow(img)
            else:
                return img
        # self.conopt_env.render(*args, **kwargs)

    @property
    def model(self):
        return self.conopt_env.world.model.model

    def step(self, action):
        env = self.conopt_env
        action = action.reshape(env.action_space.shape)

        assert env.action_space.contains(action), 'Action should be in action_space:\nSPACE=%s\nACTION=%s' % (
            env.action_space, action)

        xnext, sense = env.world.forward_dynamics(np.expand_dims(env.x, 0), np.expand_dims(action, 0))
        xnext = xnext[0]
        assert xnext.shape == env.x.shape, 'X shape changed! old=%s, new=%s' % (env.x.shape, xnext.shape)

        reward = fast_compute_cost(env.reward_fn, sense)#.compute_cost(sense)
        reward = float(np.squeeze(reward))
        # reward = float(reward[0])
        # assert(isinstance(reward, float))
        env.x = xnext
        # done = False
        # if self.horizon is not None:
        #     done = env.t >= self.horizon
        # env.t += 1
        return Step(self.convert_obs(env._get_obs()), reward, False, target_id=self.target_id)

        # rew = []
        # for i in range(env.batchsize):
        #     sense_here = {k: sense[k][i] for k in sense}
        #     reward = - fast_compute_cost(env.reward_fn, sense_here)
        #     reward = np.squeeze(reward)
        #     rew.append(reward)
        #
        # env.x = xnext
        #
        # return Step(env._get_obs(), np.squeeze(rew), False)

    @classmethod
    def shuffler(cls):
        return Shuffler()


if __name__ == "__main__":
    from sandbox.rocky.analogy.policies.conopt_particle_tracking_policy import ConoptParticleTrackingPolicy
    from rllab.sampler.utils import rollout
    while True:
        demo_seed = np.random.randint(low=0, high=1000000)
        analogy_seed = np.random.randint(low=0, high=1000000)
        target_seed = np.random.randint(low=0, high=1000000)
        demo_env = ConoptParticleEnv(seed=demo_seed, target_seed=target_seed)
        demo_policy = ConoptParticleTrackingPolicy(demo_env)
        rollout(demo_env, demo_policy, max_path_length=100, animated=True, speedup=10)
        analogy_env = ConoptParticleEnv(seed=analogy_seed, target_seed=target_seed)
        analogy_policy = ConoptParticleTrackingPolicy(analogy_env)
        rollout(analogy_env, analogy_policy, max_path_length=100, animated=True, speedup=10)
