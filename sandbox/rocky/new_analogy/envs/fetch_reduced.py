import copy
import multiprocessing
import pickle

from cached_property import cached_property

import gpr.env
from gpr_package.bin import tower_fetch_policy as tower
from rllab.core.serializable import Serializable
from rllab.envs.base import Env
from rllab.envs.gym_env import convert_gym_space
from rllab.misc import tensor_utils
from rllab.spaces import Box
from rllab.misc import logger
import numpy as np

from sandbox.rocky.new_analogy.envs.gpr_env import CachedWorldBuilder
from sandbox.rocky.new_analogy.gpr_ext.fast_forward_dynamics import FastForwardDynamics
from sandbox.rocky.s3.resource_manager import resource_manager
from sandbox.rocky.tf.envs.vec_env import VecEnv


class FetchReduced(Env, Serializable):
    def __init__(self, ref_path, k, reward_type='shaped', only_from_start=False, mocap=True):
        Serializable.quick_init(self, locals())
        task_id = tower.get_task_from_text("ab")
        expr = tower.SimFetch(nboxes=2, horizon=k, mocap=mocap, obs_type="flatten", num_substeps=1)
        self.gpr_env = expr.make(task_id)
        self.ref_path = ref_path
        self.k = k
        self.x = None
        self.reward_type = reward_type
        self.only_from_start = only_from_start
        self.mocap = mocap
        self.vec_env = VecFetchReducedEnv(self, n_envs=1)

    @cached_property
    def observation_space(self):
        return convert_gym_space(self.gpr_env.observation_space)

    @cached_property
    def action_space(self):
        # we only control the mocap position but not the orientation (which is held fixed)
        # actions for the two grippers are synced
        if self.mocap:
            return Box(low=-1, high=1, shape=(4,))
        else:
            return convert_gym_space(self.gpr_env.action_space)

    def reset(self):
        return self.vec_env.reset(dones=[True])[0]

    def step(self, action):
        obs, rewards, dones, infos = self.vec_env.step(action_n=np.asarray([action]), max_path_length=None)
        return obs[0], rewards[0], dones[0], {k: v[0] for k, v in infos.items()}

    @property
    def vectorized(self):
        return True

    def vec_env_executor(self, n_envs):
        return VecFetchReducedEnv(self, n_envs)

    def render(self):
        self.vec_env.gpr_envs[0].render()

    def unnormalize_actions(self, actions):
        if self.mocap:
            actions = np.asarray(actions)
            N = len(actions)
            return np.concatenate([
                actions[:, [0, 1, 2]],
                np.zeros((N, 1)),
                np.ones((N, 1)) * 1.57,
                np.zeros((N, 1)),
                actions[:, [3, 3]],
            ], axis=1)
        else:
            return np.asarray(actions)

    def log_diagnostics(self, paths):
        logger.record_tabular_misc_stat('FinalReward', np.asarray([p["rewards"][-1] for p in paths]))


class VecFetchReducedEnv(VecEnv):
    def __init__(self, env: FetchReduced, n_envs):
        gpr_env = env.gpr_env
        gpr_envs = [
            gpr.env.Env(world_builder=CachedWorldBuilder(gpr_env), reward=gpr_env.reward,
                        horizon=gpr_env.horizon, task_id=gpr_env.task_id, delta_reward=gpr_env.delta_reward,
                        delta_obs=gpr_env.delta_obs)
            for _ in range(n_envs)
            ]
        self.env = env
        self.gpr_envs = gpr_envs
        self.fast_forward_dynamics = FastForwardDynamics(
            env=gpr_env,
            n_parallel=multiprocessing.cpu_count(),
            extra_specs={"use_site_xpos"}
        )
        self.n_envs = n_envs
        self.final_xpos = np.zeros((n_envs, gpr_env.world.dimxpos))
        self.site_xpos = np.zeros((n_envs, gpr_env.world.dimxpos))
        self.xids = np.zeros((n_envs), dtype=np.int)
        self.ts = np.zeros((n_envs), dtype=np.int)
        # self.xs = np.zeros((n_envs, gpr_env.world.dimx))

    def reset(self, dones, seeds=None, *args, **kwargs):
        dones = np.cast['bool'](dones)
        n_dones = np.sum(dones)
        if n_dones == 0:
            return []
        assert seeds is None
        xs = self.env.ref_path["env_infos"]["x"]
        site_xpos = self.env.ref_path["env_infos"]["site_xpos"]
        if self.env.only_from_start:
            xids = np.zeros(n_dones, dtype=np.int)
        else:
            xids = np.random.randint(low=0, high=max(1, len(xs) - self.env.k), size=n_dones)
        self.final_xpos[dones] = site_xpos[np.minimum(xids + self.env.k, len(xs) - 1)]
        self.site_xpos[dones] = site_xpos[xids]
        self.xids[dones] = xids
        self.ts[dones] = 0
        obs = []
        done_idx = 0
        for gpr_env, done in zip(self.gpr_envs, dones):
            if done:
                gpr_env.reward.reset()
                gpr_env.x = xs[xids[done_idx]]
                done_idx += 1
                obs.append(copy.deepcopy(gpr_env._get_obs()[0]))
        return obs

    def step(self, action_n, max_path_length):
        action_n = self.env.unnormalize_actions(action_n)
        action_n = action_n.astype(np.float64)

        xs = np.asarray([env.x for env in self.gpr_envs])

        prev_site_xpos = self.site_xpos
        prev_dist = np.linalg.norm(prev_site_xpos - self.final_xpos, axis=1)
        xnext, rewards, senses = self.fast_forward_dynamics(xs, action_n)
        after_site_xpos = senses["site_xpos"]
        after_dist = np.linalg.norm(after_site_xpos - self.final_xpos, axis=1)

        self.site_xpos = np.copy(after_site_xpos)

        out_obs = []
        out_rewards = []
        out_dones = []
        out_infos = []
        for idx, (xnext_i, gpr_env, action_i, reward_i) in enumerate(zip(xnext, self.gpr_envs, action_n, rewards)):
            gpr_env.x = xnext_i
            reward_i = float(reward_i)
            if self.env.reward_type == 'shaped':
                if self.ts[idx] == 0:
                    reward_i = - after_dist[idx]
                else:
                    reward_i = prev_dist[idx] - after_dist[idx]
            done = False

            obs, diverged = gpr_env._get_obs()
            if diverged:
                done = True
                reward_i = -1000

            if gpr_env.runner is not None:
                gpr_env.runner.reward = "%.2f" % reward_i
                gpr_env.runner.action = action_i

            out_obs.append(obs)
            out_rewards.append(reward_i)
            out_dones.append(done)
            out_infos.append({"diverged": diverged})

        out_dones = np.asarray(out_dones)
        self.ts += 1
        if max_path_length is not None:
            out_dones[self.ts >= max_path_length] = True

        return out_obs, out_rewards, out_dones, tensor_utils.stack_tensor_dict_list(out_infos)


if __name__ == "__main__":
    path = pickle.load(open(resource_manager.get_file("fetch_single_traj"), "rb"))
    env = FetchReduced(ref_path=path, k=1, reward_type='original')

    task_id = tower.get_task_from_text("ab")
    expr = tower.SimFetch(nboxes=2, horizon=1e9, mocap=True, obs_type="full_state", num_substeps=1)
    ref_env = expr.make(task_id)

    env.reset()
    xid = env.vec_env.xids[0]
    x = path["env_infos"]["x"][xid]
    ref_env.reset_to(x)

    action = [0.1, 0.1, 0.1, 1.]

    ob, reward, _, _ = env.step(action)
    ob_, reward_, _, _ = ref_env.step(env.unnormalize_actions([action])[0])
    for o1, o2 in zip(ob, ob_):
        np.testing.assert_allclose(o1, o2)
    np.testing.assert_allclose(reward, reward_)

    env = FetchReduced(ref_path=path, k=1, reward_type='shaped')
    env.reset()
    xid = env.vec_env.xids[0]
    x = path["env_infos"]["x"][xid]
    ref_env.reset_to(x)

    prev_site_xpos = env.vec_env.site_xpos[0]
    tot_reward = 0.
    tot_reward += env.step(action)[1]
    tot_reward += env.step(action)[1]
    tot_reward += env.step(action)[1]
    after_site_xpos = env.vec_env.site_xpos[0]
    final_site_xpos = env.vec_env.final_xpos[0]

    should_reward = np.linalg.norm(prev_site_xpos - final_site_xpos) - np.linalg.norm(after_site_xpos - final_site_xpos)

    np.testing.assert_allclose(should_reward, tot_reward)
