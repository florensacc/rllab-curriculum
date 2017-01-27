import copy
import glob
import importlib
import multiprocessing
import os
import random

from cached_property import cached_property
from gym.spaces import prng
from gym.utils import seeding

from rllab.core.serializable import Serializable
from rllab.envs.base import Env
from rllab.envs.gym_env import convert_gym_space
import gpr.env
import gpr.envs
from rllab.envs.base import Step
import numpy as np
from rllab.misc import logger
from rllab.misc.ext import using_seed
from sandbox.rocky.new_analogy.gpr_ext.fast_forward_dynamics import FastForwardDynamics
from sandbox.rocky.tf.envs.vec_env import VecEnv
from sandbox.rocky.tf.envs.vec_env_executor import VecEnvExecutor
import logging

from sandbox.rocky.tf.misc import tensor_utils


def load_with_args(name, kwargs):
    name = name.replace("/", ".")
    try:
        mod = importlib.import_module('gpr.envs.%s' % name)
        return mod.Experiment(**kwargs)
    except ImportError:
        choices = glob.glob('%s/%s*.py' % (os.path.dirname(__file__), name))
        if len(choices) == 1:
            modname = os.path.splitext(os.path.basename(choices[0]))[0]
            mod = importlib.import_module('gpr.envs.%s' % modname)
            return mod.Experiment(**kwargs)
        elif len(choices) == 0:
            raise
        else:
            raise Exception('Ambiguous experiment name %s -- choices: %s' % (name, choices))


class GprEnv(Env, Serializable):
    def __init__(
            self,
            env_name,
            gpr_env=None,
            experiment_args=None,
            make_args=None,
            xinits=None,
            seed=None,
            task_id=None,
            stage=None
    ):
        Serializable.quick_init(self, locals())
        self.env_name = env_name
        self.task_id = task_id
        self.seed = seed
        self.stage = stage
        with using_seed(seed):
            if seed is not None:
                env_seed = seed
            else:
                env_seed = np.random.randint(low=0, high=np.iinfo(np.int32).max)
            if gpr_env is None:
                if experiment_args is None:
                    experiment_args = dict()
                expr = load_with_args(env_name, experiment_args)
                if make_args is None:
                    make_args = dict()
                if task_id is not None:
                    gpr_env = expr.make(task_id, **make_args)
                elif hasattr(expr, "task_id_iterator"):
                    task_id = np.random.choice(list(expr.task_id_iterator()))
                    gpr_env = expr.make(task_id, **make_args)
                else:
                    gpr_env = expr.make(**make_args)
            self.gpr_env = gpr_env
            gpr_env.seed(env_seed)
            self.xinits = xinits
            self.reset()

    @cached_property
    def observation_space(self):
        return convert_gym_space(self.gpr_env.observation_space)

    @cached_property
    def action_space(self):
        return convert_gym_space(self.gpr_env.action_space)

    def reset(self):
        if self.xinits is not None:
            xinit = random.choice(self.xinits)
            xinit = xinit[:self.gpr_env.world.dimx]
            return copy.deepcopy(self.gpr_env.reset_to(xinit))
        if self.seed is None:
            seed = np.random.randint(low=0, high=np.iinfo(np.int32).max)
        else:
            seed = self.seed
        self.gpr_env.seed(seed)
        prng.seed(seed)
        return copy.deepcopy(self.gpr_env.reset())

    def step(self, action):
        info = dict(x=np.copy(self.gpr_env.x))
        try:
            action = np.asarray(action, dtype=np.float)
            obs, rew, done, _ = self.gpr_env.step(action)
            assert np.max(np.abs(self.observation_space.flatten(obs))) < 1000
        except AssertionError:
            # wtf...
            obs = self.observation_space.default_value
            rew = 0  # -1000
            done = True
        if hasattr(self.gpr_env, "task_id_iterator"):
            info["task_id"] = self.gpr_env.task_id
        return Step(observation=copy.deepcopy(obs), reward=rew, done=done, **info)

    def render(self, mode='human', close=False):
        return self.gpr_env.render(mode=mode, close=close)

    def _is_success(self, path):
        if self.env_name in ["stack", "fetch_bc"]:
            from sandbox.rocky.new_analogy import fetch_utils
            return len(fetch_utils.find_stageinit_points(self, path)) == len(self.gpr_env.task_id[0])
            # model = self.gpr_env.world.model
            # site_names = model.site_names
            # n_sites = len(site_names)
            # site_xpos = path["observations"][:, :n_sites * 3]
            #
            # geoms = sorted([name for name in site_names if name.startswith("geom")])
            # success = True
            #
            # for geom, next_geom in zip(geoms, geoms[1:]):
            #     geom0_idx = site_names.index(geom)
            #     geom1_idx = site_names.index(next_geom)
            #
            #     geom0_xpos = site_xpos[:, geom0_idx * 3:geom0_idx * 3 + 3]
            #     geom1_xpos = site_xpos[:, geom1_idx * 3:geom1_idx * 3 + 3]
            #     dist = np.linalg.norm(geom0_xpos - geom1_xpos, axis=1)
            #     if dist[-1] > 0.06:
            #         success = False
            #         break
            # return success
        else:
            raise NotImplementedError

    def _is_stage_success(self, path, stage):
        from sandbox.rocky.new_analogy import fetch_utils
        return len(fetch_utils.find_stageinit_points(self, path)) >= stage + 2

        # model = self.gpr_env.world.model
        # site_names = model.site_names
        # n_sites = len(site_names)
        # geoms = sorted([name for name in site_names if name.startswith("geom")])
        # site_xpos = path["observations"][:, :n_sites * 3]
        #
        # for geom, next_geom in list(zip(geoms, geoms[1:]))[:stage + 1]:
        #     geom0_idx = site_names.index(geom)
        #     geom1_idx = site_names.index(next_geom)
        #
        #     geom0_xpos = site_xpos[:, geom0_idx * 3:geom0_idx * 3 + 3]
        #     geom1_xpos = site_xpos[:, geom1_idx * 3:geom1_idx * 3 + 3]
        #     dist = np.linalg.norm(geom0_xpos - geom1_xpos, axis=1)
        #     if np.min(dist) > 0.06:
        #         return False
        # return True

    def log_diagnostics(self, paths):
        if self.env_name == "stack":
            rate = np.mean([self._is_success(path) for path in paths])
            logger.record_tabular('SuccessRate', rate)
        elif self.env_name == "fetch_bc":
            n_boxes = len(self.gpr_env.task_id[0])
            if paths[0]["env_infos"]["stagewise"][0]:
                init_stages = np.asarray([p["env_infos"]["stage"][0] for p in paths])
                max_stages = np.asarray([np.max(p["env_infos"]["stage"]) for p in paths])
                rate = np.mean(np.greater(max_stages, init_stages))
                logger.record_tabular('StageSuccessRate', rate)
                for stage in range(n_boxes - 1):
                    if np.any(np.equal(init_stages, stage)):
                        rate = np.sum(
                            np.logical_and(np.equal(init_stages, stage), np.greater(max_stages, init_stages))) / \
                               np.sum(np.equal(init_stages, stage))
                    else:
                        rate = np.nan
                    logger.record_tabular('StageSuccessRate(Stage{})'.format(stage), rate)
            else:
                rate = np.mean([self._is_success(path) for path in paths])
                logger.record_tabular('SuccessRate', rate)
                for stage in range(n_boxes - 1):
                    rate = np.mean([self._is_stage_success(path, stage) for path in paths])
                    logger.record_tabular('SuccessRate(Stage{})'.format(stage), rate)
        else:
            if self.env_name in ["I1_copter_3_targets", "TF2", "tower", "fetch.sim_fetch"]:
                success_threshold = 4
            else:
                raise NotImplementedError("Unknown env: %s" % self.env_name)
            if "raw_rewards" in paths[0]:
                logger.record_tabular('SuccessRate',
                                      np.mean([p["raw_rewards"][-1] >= success_threshold for p in paths]))
            else:
                logger.record_tabular('SuccessRate', np.mean([p["rewards"][-1] >= success_threshold for p in paths]))
            if self.env_name in ["tower", "fetch.sim_fetch"]:
                logger.record_tabular_misc_stat('FinalReward', np.asarray([p["rewards"][-1] for p in paths]))

    def vec_env_executor(self, n_envs):
        return VecGprEnv(self, n_envs)

    @property
    def vectorized(self):
        return True


class CachedWorldBuilder(object):
    def __init__(self, gpr_env):
        self.gpr_env = gpr_env

    def to_world(self, seed=0):
        return self.gpr_env.world

    @property
    def world_params(self):
        return self.gpr_env.world_builder.world_params


class VecGprEnv(VecEnvExecutor):
    def __init__(self, env, n_envs):
        gpr_env = env.gpr_env
        envs = [
            GprEnv(
                env_name=env.env_name,
                gpr_env=gpr.env.Env(world_builder=CachedWorldBuilder(gpr_env), reward=gpr_env.reward,
                                    horizon=gpr_env.horizon, task_id=gpr_env.task_id, delta_reward=gpr_env.delta_reward,
                                    delta_obs=gpr_env.delta_obs, clip_reward=gpr_env.clip_reward,
                                    max_reward=gpr_env.max_reward),
                task_id=env.task_id,
                xinits=env.xinits,
                seed=env.seed,
                stage=env.stage,
            )
            for _ in range(n_envs)
            ]
        VecEnvExecutor.__init__(self, envs)

        self.fast_forward_dynamics = FastForwardDynamics(env=gpr_env, n_parallel=multiprocessing.cpu_count())
        self.noise_levels = np.zeros(self.n_envs)
        self.stagewise = False
        self.xinits = None
        self.prev_stages = np.zeros(self.n_envs)

    def reset(self, dones, seeds=None, *args, **kwargs):
        dones = np.cast['bool'](dones)
        if seeds is None:
            seeds = [None] * self.n_envs
        results = []
        for idx, (env, seed, done) in enumerate(zip(self.envs, seeds, dones)):
            if done:
                env.seed = seed
                env.xinits = self.xinits
                results.append(env.reset())
        self.ts[dones] = 0
        self.prev_stages[dones] = 0
        return results

    def inject_noise(self, noise_levels):
        if noise_levels is None or len(noise_levels) == 0:
            self.noise_levels = np.zeros(self.n_envs)
        else:
            self.noise_levels = np.random.choice(noise_levels, size=self.n_envs)

    def set_stagewise(self, stagewise):
        self.stagewise = stagewise

    def set_xinits(self, xinits):
        self.xinits = xinits

    def step(self, action_n, max_path_length):
        action_n = action_n.astype(np.float64)
        action_n = action_n + self.noise_levels[:, None] * np.random.randn(*action_n.shape)
        gpr_env = self.envs[0].gpr_env
        for action in action_n:
            assert gpr_env.action_space.contains(action), \
                'Action should be in action_space:\nSPACE=%s\nACTION=%s' % (gpr_env.action_space, action)

        xs = np.asarray([env.gpr_env.x for env in self.envs])

        xnext, rewards, senses, obs_list, diverged_list = self.fast_forward_dynamics(xs, action_n, get_obs=True)

        # r = [None]
        # # coparison
        # def lambda_over_sense(s, i):
        #     r[0] = gpr_env.reward.compute_reward(s)
        # gpr_env.world.forward_dynamics(xs, action_n, lambda_over_sense=lambda_over_sense)
        # print(r[0][0])
        # print(rewards[0])
        # import ipdb; ipdb.set_trace()
        # assert r[0] == rewards[0]
        # exit()

        out_obs = []
        out_rewards = []
        out_dones = []
        out_infos = []

        from sandbox.rocky.new_analogy import fetch_utils
        stages = fetch_utils.compute_stage(self.envs[0], senses['site_xpos'])

        for idx, (xnext_i, env, action_i, reward_i, obs_i, diverged_i) in \
                enumerate(zip(xnext, self.envs, action_n, rewards, obs_list, diverged_list)):
            gpr_env = env.gpr_env
            assert xnext_i.shape == gpr_env.x.shape, 'X shape changed! old=%s, new=%s' % (gpr_env.x.shape,
                                                                                          xnext_i.shape)

            if hasattr(gpr_env, 'traj'):  # recording trajectories
                gpr_env.traj.solution["x"][gpr_env.t] = gpr_env.x
                gpr_env.traj.solution["u"][gpr_env.t] = action_i

            gpr_env.x = xnext_i
            reward_i = float(reward_i)
            assert (isinstance(reward_i, float))
            done = False
            gpr_env.t += 1

            #### Special case for fetch bc environment, to terminate early when blocks fall off the table
            if env.env_name == "fetch_bc":
                diverged_i = diverged_i or self._check_box_off_table(
                    gpr_env=gpr_env,
                    current_site_xpos=senses['site_xpos'][idx],
                )

            if gpr_env.horizon is not None:
                done = gpr_env.t >= gpr_env.horizon
            if not done and max_path_length is not None:
                done = gpr_env.t >= max_path_length

            if self.stagewise and gpr_env.t > 1 and stages[idx] > self.prev_stages[idx]:
                # finish as soon as advanced to new stage
                done = True
            self.prev_stages[idx] = stages[idx]

            if diverged_i:
                logging.info('obs threshold exceeded')
                done = True
                reward_i = gpr_env.world.params.reward_at_divergence
                if reward_i is None:
                    reward_i = -10
            # apply the reward clipping logic regardless
            if gpr_env.clip_reward:
                if gpr_env.t == 1:
                    gpr_env.min_reward = reward_i
                else:
                    if gpr_env.max_reward:
                        gpr_env.min_reward = max(gpr_env.min_reward, reward_i)
                    reward_i = max(gpr_env.min_reward, reward_i)
            if gpr_env.delta_reward:
                reward_i -= gpr_env.last_reward
                gpr_env.last_reward += reward_i
            if gpr_env.runner is not None:
                gpr_env.runner.reward = "%.2f" % reward_i
                gpr_env.runner.action = action_i

            if hasattr(gpr_env, 'traj'):  # recording trajectories
                gpr_env.traj.solution["reward"][gpr_env.t - 1] = reward_i
                if done and gpr_env.n_resets % gpr_env.save_frequency == 0:
                    gpr_env.save_trajectory('%s/trajectory%09d.pkl' % (gpr_env.path, gpr_env.n_resets))

            out_obs.append(obs_i)
            out_rewards.append(reward_i)
            out_dones.append(done)
            out_infos.append({
                "diverged": diverged_i,
                "x": xnext_i,
                "site_xpos": senses["site_xpos"][idx],
                "noise_level": self.noise_levels[idx],
                "stage": stages[idx],
                "stagewise": self.stagewise,
            })

        return out_obs, out_rewards, out_dones, tensor_utils.stack_tensor_dict_list(out_infos)

    def _check_box_off_table(self, gpr_env, current_site_xpos):
        # return False
        site_names = gpr_env.world.model.site_names
        for idx, site_name in enumerate(site_names):
            if site_name.startswith('geom'):
                cur_geom_xpos = current_site_xpos[idx * 3:idx * 3 + 3]
                # check coordinate
                if not (-0.1 <= cur_geom_xpos[0] <= 0.7 and
                                    -0.4 <= cur_geom_xpos[1] <= 0.4 and
                                    0.4 <= cur_geom_xpos[2] <= 0.8):
                    return True
        return False
