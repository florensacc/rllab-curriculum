import itertools
import random

import numpy as np
import pyprind
import tensorflow as tf
from gym.spaces import prng

from gpr.core import ObservationParams
from gpr.utils.rotation import mat2euler_batch, euler2mat_batch
from gpr import Trajectory
from gpr.envs import fetch_rl
from gpr.envs import fetch_bc
import gpr.env
from rllab.core.serializable import Serializable
from rllab.envs.base import Env
from rllab.envs.env_spec import EnvSpec
from rllab.envs.proxy_env import ProxyEnv
from rllab.misc import logger
from rllab.misc.ext import AttrDict
from rllab.misc.special import weighted_sample_n
from rllab.sampler import parallel_sampler
from rllab.sampler.stateful_pool import singleton_pool
from rllab.sampler.utils import rollout
from sandbox.rocky.new_analogy.envs.gpr_env import GprEnv, gpr, VecGprEnv
import gpr_package.bin.tower_fetch_policy as tower
from sandbox.rocky.tf.core.parameterized import Parameterized
from sandbox.rocky.tf.distributions.diagonal_gaussian import DiagonalGaussian
from sandbox.rocky.tf.envs.base import TfEnv, VecTfEnv
from sandbox.rocky.tf.envs.vec_env import VecEnv
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.policies.base import Policy
from sandbox.rocky.tf.spaces import Discrete, Product
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler


def fetch_env(horizon=1000, height=2, seed=None, usage="prescribed", task_id=None):
    if usage == "prescribed":
        env = TfEnv(
            GprEnv(
                "fetch_bc",
                experiment_args=dict(horizon=horizon),
                make_args=dict(height=height, task_id=task_id),
                seed=seed
            )
        )
    elif usage == "rl":
        env = TfEnv(
            GprEnv(
                "fetch_bc",
                experiment_args=dict(horizon=horizon),
                make_args=dict(delta_reward=True, height=height, task_id=task_id),
                seed=seed
            )
        )
    else:
        raise NotImplementedError
    return env


def gpr_fetch_env(horizon=1000, height=2):
    expr = fetch_bc.Experiment(horizon=horizon)
    env = expr.make(height=height)
    return env


def gpr_fetch_expr(horizon=1000, usage="prescribed"):
    if usage == "prescribed":
        expr = fetch_bc.Experiment(horizon=horizon)
    elif usage in ["pi2", "rl"]:
        expr = fetch_rl.Experiment(horizon=horizon)
    return expr


def get_gpr_env(env):
    if isinstance(env, gpr.env.Env):
        return env
    elif isinstance(env, ProxyEnv):
        return get_gpr_env(env.wrapped_env)
    elif isinstance(env, GprEnv):
        return env.gpr_env
    else:
        import ipdb;
        ipdb.set_trace()


def get_vec_gpr_env(vec_env):
    if isinstance(vec_env, VecTfEnv):
        return get_vec_gpr_env(vec_env.vec_env)
    elif isinstance(vec_env, VecGprEnv):
        return vec_env
    else:
        import ipdb;
        ipdb.set_trace()


class RelativeFetchPolicy(Serializable):
    def __init__(self, env):
        Serializable.quick_init(self, locals())
        gpr_env = get_gpr_env(env)
        policy = tower.FetchPolicy(gpr_env.task_id)
        self.wrapped_get_action = absolute2relative_wrapper(policy.get_action)
        self.env = env
        self.gpr_env = gpr_env
        self.abs_policy = policy
        self.vec_policy = None

    def get_action(self, ob):
        # check!
        ob_ = self.gpr_env.world.observe(self.gpr_env.x)[0]
        assert np.all(np.equal(ob, ob_))
        action = self.wrapped_get_action(self.gpr_env)
        return action, dict()

    def inform_vec_env(self, vec_env):
        if self.vec_policy is None:
            self.vec_policy = VecRelativeFetchPolicy(vec_env, self.abs_policy)
        else:
            self.vec_policy.vec_env = vec_env

    def get_actions(self, observations):
        # if len(observations) == 1:
        #     return self.get_action(observations[0])[0].reshape((1, -1)), dict()
        return self.vec_policy.get_actions(observations)

    def reset(self, dones=None):
        pass


def absolute2relative_wrapper(policy):
    def get_action(env):
        world = env.world
        assert world.nmocap == 1
        assert world.dimu == 8

        params_copy = world.params
        world.params = world.params._replace(observation=ObservationParams(flatten=False,
                                                                           qpos=False,
                                                                           qvel=False,
                                                                           qpos_robot=True,
                                                                           qvel_robot=True,
                                                                           site_xpos=True,
                                                                           sites_relative_to=None,
                                                                           skip_sites=1))
        obs = world.observe(env.x)[0]
        env.world.params = params_copy
        abs_action = policy(obs).reshape(1, 8)
        mocap, ctrl = np.split(np.copy(abs_action), [world.nmocap * 6], axis=-1)

        # gripper
        assert ctrl[:, 0] == ctrl[:, 1]

        # mocap
        mocap_xpos, mocap_euler = np.split(np.copy(mocap), 2, axis=1)
        gripper_xpos, gripper_xmat = world.get_relative_frame(env.x.reshape(1, -1))
        gripper_xmat_inv = np.linalg.inv(gripper_xmat)

        mocap_xpos = np.matmul(gripper_xmat_inv, (mocap_xpos - gripper_xpos).reshape(-1, 3, 1)).reshape(-1, 3)
        mocap_euler = mat2euler_batch(np.matmul(gripper_xmat_inv, euler2mat_batch(mocap_euler)))

        mocap = np.concatenate((mocap_xpos, mocap_euler), axis=-1)

        if world.params.action.mocap_fix_orientation:
            mocap[:, 3:] = 0

        # test
        rel_action = np.concatenate((mocap, ctrl), axis=-1)
        preprocessed_rel_action = np.concatenate(world.preprocess_action(env.x.reshape(1, -1), rel_action), axis=-1)
        assert np.linalg.norm(preprocessed_rel_action - abs_action) < 1e-3

        return rel_action[0]

    return get_action


class VecRelativeFetchPolicy(Serializable):
    def __init__(self, vec_env, abs_policy):
        self.vec_env = vec_env
        self.abs_policy = abs_policy

    def get_actions_from_xs(self, xs):
        vec_gpr_env = self.vec_env.vec_env
        assert isinstance(vec_gpr_env, VecGprEnv)
        model = vec_gpr_env.fast_forward_dynamics.mjparallel.model

        dimq = model.nq
        qpos = xs[..., :dimq]
        qvel = xs[..., dimq:]
        qpos = np.asarray(qpos, dtype=np.float64, order='C')
        qvel = np.asarray(qvel, dtype=np.float64, order='C')
        vec_gpr_env = self.vec_env.vec_env
        world = vec_gpr_env.fast_forward_dynamics.env.world

        params_copy = world.params
        world.params = world.params._replace(observation=ObservationParams(flatten=False,
                                                                           qpos=False,
                                                                           qvel=False,
                                                                           qpos_robot=True,
                                                                           qvel_robot=True,
                                                                           site_xpos=True,
                                                                           sites_relative_to=None,
                                                                           skip_sites=1))
        full_state_obs, _ = vec_gpr_env.fast_forward_dynamics.get_obs(qpos=qpos, qvel=qvel)
        world.params = params_copy

        abs_actions = []
        for o in full_state_obs:
            # o = (o[0], o[1], o[2][3:])
            abs_actions.append(self.abs_policy.get_action(o))
        abs_actions = np.asarray(abs_actions)

        mocap, ctrl = np.split(abs_actions, [world.nmocap * 6], axis=-1)

        # gripper
        assert np.allclose(ctrl[..., 0], ctrl[..., 1])

        # mocap
        mocap_xpos, mocap_euler = np.split(np.copy(mocap), 2, axis=-1)

        origin_xpos, origin_xmat = vec_gpr_env.fast_forward_dynamics.get_relative_frame(qpos, qvel)

        origin_xmat_inv = np.linalg.inv(origin_xmat)

        mocap_xpos = np.matmul(origin_xmat_inv, (mocap_xpos - origin_xpos).reshape(-1, 3, 1)).reshape(-1, 3)
        mocap_euler = mat2euler_batch(np.matmul(origin_xmat_inv, euler2mat_batch(mocap_euler)))

        mocap = np.concatenate((mocap_xpos, mocap_euler), axis=-1)

        if world.params.action.mocap_fix_orientation:
            mocap[:, 3:] = 0

        # test
        rel_action = np.concatenate((mocap, ctrl), axis=-1)
        preprocessed_rel_action = np.concatenate(
            vec_gpr_env.fast_forward_dynamics.preprocess_action(qpos, qvel, rel_action), axis=-1)

        assert np.all(np.linalg.norm(preprocessed_rel_action - abs_actions, axis=-1) < 1e-3)

        return rel_action, dict()

    def get_actions(self, observations):
        vec_gpr_env = self.vec_env.vec_env
        assert isinstance(vec_gpr_env, VecGprEnv)
        assert len(observations) == vec_gpr_env.n_envs
        model = vec_gpr_env.fast_forward_dynamics.mjparallel.model

        # first, check that the observations match
        xs = np.asarray([env.gpr_env.x for env in self.vec_env.vec_env.envs])
        dimq = model.nq
        qpos = xs[..., :dimq]
        qvel = xs[..., dimq:]
        qpos = np.asarray(qpos, dtype=np.float64, order='C')
        qvel = np.asarray(qvel, dtype=np.float64, order='C')
        check_obs, _ = vec_gpr_env.fast_forward_dynamics.get_obs(qpos=qpos, qvel=qvel)

        assert np.allclose(check_obs, observations)

        world = vec_gpr_env.fast_forward_dynamics.env.world
        assert world.params.observation.sites_relative_to is not None
        assert world.nmocap == 1
        assert world.dimu == 8

        return self.get_actions_from_xs(xs)


def fetch_prescribed_policy(env):
    gpr_env = get_gpr_env(env)
    assert gpr_env.world.params.observation.sites_relative_to is not None
    return RelativeFetchPolicy(env)


def fetch_discretized_prescribed_policy(env, disc_intervals):
    gpr_env = get_gpr_env(env)
    assert gpr_env.world.params.observation.sites_relative_to is not None
    return DiscretizedRelativeFetchPolicy(RelativeFetchPolicy(env), disc_intervals)


disc_intervals = np.asarray([
    [-0.3, -0.1, -0.03, -0.01, -0.003, -0.001, -0.0003, 0, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
    [-0.3, -0.1, -0.03, -0.01, -0.003, -0.001, -0.0003, 0, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
    [-0.3, -0.1, -0.03, -0.01, -0.003, -0.001, -0.0003, 0, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
])


class DiscretizedRelativeFetchPolicy(object):
    def __init__(self, rel_policy, disc_intervals):
        self.rel_policy = rel_policy
        self.disc_intervals = list(map(np.asarray, disc_intervals))

    @property
    def vectorized(self):
        return True

    def reset(self, dones=None):
        self.rel_policy.reset(dones)

    def inform_vec_env(self, vec_env):
        if hasattr(self.rel_policy, 'inform_vec_env'):
            self.rel_policy.inform_vec_env(vec_env)

    def get_actions(self, observations):
        actions, agent_infos = self.rel_policy.get_actions(observations)
        return self.discretize_actions(actions), dict(agent_infos, original_action=actions)

    def discretize_actions(self, actions):
        bins_0 = self.disc_intervals[0]
        bins_1 = self.disc_intervals[1]
        bins_2 = self.disc_intervals[2]
        actions = np.asarray(actions, dtype=np.float)
        new_actions = np.array(actions, dtype=np.float)
        # only discretize the first 3 dimensions

        for disc_id in range(3):
            # the logic is that discretization should not increase the magnitude of the action.
            bins = self.disc_intervals[disc_id]
            cur_actions = actions[:, disc_id]

            shrinkage_mask = (np.abs(cur_actions)[:, None] >= np.abs(bins)[None, :])
            # lp = np.max(bins[bins<0]) # get smallest precision for < 0
            # up = np.min(bins[bins>0]) # get smallest precision for > 0
            # lp_idx = np.where(bins == lp)[0][0]
            # up_idx = np.where(bins == up)[0][0]
            # if no shrinkage, assign it to the
            bin_ids = np.argmax(1. / (np.abs(cur_actions[:, None] - bins[None, :]) + 1e-8) * (1.0 * shrinkage_mask),
                                axis=1)
            # has_shrinkage_mask = np.any(shrinkage_mask, axis=1)
            new_actions[:, disc_id] = bins[bin_ids]  # [has_shrinkage_mask]]
            # new_actions[-1e-8>cur_actions>=lp, disc_id] = lp
            # new_actions[1e-8<cur_actions<=up, disc_id] = up
            # new_actions[-1e-8<=cur_actions<=1e-8, disc_id] = 0

            # no_shrinkage_mask = np.logical_not(np.any(shrinkage_mask, axis=1))
            # new_actions[np.logical_and(no_shrinkage_mask, new_actions[:,disc_id]<0)] =
            # bin_ids = np.abs(cur_actions)[:, None] <= np.abs(bins)[None, :]
            # new_actions[:, disc_id]
            # import ipdb; ipdb.set_trace()
        new_actions[:, 0] = bins_0[np.argmin(
            np.abs(actions[:, 0][:, None] - bins_0[None, :]),
            axis=1
        )]
        new_actions[:, 1] = bins_1[np.argmin(
            np.abs(actions[:, 1][:, None] - bins_1[None, :]),
            axis=1
        )]
        new_actions[:, 2] = bins_2[np.argmin(
            np.abs(actions[:, 2][:, None] - bins_2[None, :]),
            axis=1
        )]
        return new_actions

    def get_action(self, observation):
        action, agent_infos = self.rel_policy.get_action(observation)
        return self.discretize_actions([action])[0], dict(agent_infos, original_action=action)


class NoisyEnvWrapper(ProxyEnv, Serializable):
    def __init__(self, wrapped_env, noise_levels=None):
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, wrapped_env)
        self.noise_levels = noise_levels
        self.current_noise_level = None

    def reset(self):
        if self.noise_levels is not None:
            self.current_noise_level = random.choice(self.noise_levels)
        return self.wrapped_env.reset()

    def step(self, action):
        if self.noise_levels is not None:
            action = action + self.current_noise_level * np.random.randn(*self.action_space.shape)
        return self.wrapped_env.step(action)


def demo_path(seed=0, env=None, policy=None, noise_levels=None, animated=False, speedup=10):
    if env is None:
        env = fetch_env()
    if policy is None:
        policy = fetch_prescribed_policy(env)
    policy.env = env
    env = NoisyEnvWrapper(env, noise_levels=noise_levels)
    gpr_env = get_gpr_env(env)
    gpr_env.seed(seed)
    prng.seed(seed)
    return rollout(env=env, agent=policy, max_path_length=gpr_env.horizon, animated=animated,
                   speedup=speedup)


def _worker_collect_path(G, seed, noise_levels):
    env = G.env
    policy = G.policy
    return demo_path(seed=seed, env=env, policy=policy, noise_levels=noise_levels)


def demo_paths(seeds, env=None, policy=None, show_progress=True, noise_levels=None):
    if env is None:
        env = fetch_env()
    if policy is None:
        policy = fetch_prescribed_policy(env)
    parallel_sampler.populate_task(env=env, policy=policy)
    paths = []
    if show_progress:
        progbar = pyprind.ProgBar(len(seeds))
    for path in singleton_pool.run_imap_unordered(
            _worker_collect_path,
            [(x, noise_levels) for x in seeds]
    ):
        paths.append(path)
        if show_progress:
            progbar.update()
    if show_progress:
        if progbar.active:
            progbar.stop()
    return paths


def _worker_init_tf(G):
    if not hasattr(G, 'sess') or G.sess is None:
        G.sess = tf.Session()
        G.sess.__enter__()


def annotate_paths(paths):
    xs = np.concatenate([p["env_infos"]["x"] for p in paths], axis=0)
    sampler = list(_cached_sampler.values())[0]
    vec_gpr_env = sampler.vec_env.vec_env
    assert isinstance(vec_gpr_env, VecGprEnv)
    abs_policy = tower.FetchPolicy(vec_gpr_env.envs[0].gpr_env.task_id)
    vec_policy = VecRelativeFetchPolicy(vec_env=sampler.vec_env, abs_policy=abs_policy)
    actions = vec_policy.get_actions_from_xs(xs)[0]
    path_lens = np.asarray([len(p["actions"]) for p in paths])
    path_actions = np.split(actions, np.cumsum(path_lens)[:-1], axis=0)
    for p, p_actions in zip(paths, path_actions):
        p["actions"] = p_actions


def _worker_annotate(G, idx, path):
    env = G.env
    gpr_env = get_gpr_env(env)
    demo_policy = fetch_prescribed_policy(gpr_env)
    obs = env.observation_space.unflatten_n(path["observations"])
    actions = []
    for x, ob in zip(path["env_infos"]["x"], obs):
        gpr_env.reset_to(x)
        demo_policy.env = gpr_env
        actions.append(demo_policy.get_action(ob)[0])
    return idx, np.asarray(actions)


def policy_paths(seeds, policy, env=None, show_progress=True, noise_levels=None):
    if env is None:
        env = fetch_env()
    if singleton_pool.n_parallel > 1:
        singleton_pool.run_each(_worker_init_tf)
    parallel_sampler.populate_task(env=env, policy=policy)
    policy_params = policy.get_param_values()
    scope = None
    singleton_pool.run_each(
        parallel_sampler._worker_set_policy_params,
        [(policy_params, scope)] * singleton_pool.n_parallel
    )
    paths = []
    if show_progress:
        progbar = pyprind.ProgBar(len(seeds))
    for path in singleton_pool.run_imap_unordered(
            _worker_collect_path,
            [(x, noise_levels) for x in seeds]
    ):
        paths.append(path)
        if show_progress:
            progbar.update()
    if show_progress:
        if progbar.active:
            progbar.stop()
    return paths


_cached_sampler = dict()


def new_policy_paths(
        seeds, policy, env=None, show_progress=True, noise_levels=None, horizon=None,
        stagewise=False, xinits=None):
    if env is None:
        env = fetch_env()
    gpr_env = get_gpr_env(env)
    if singleton_pool.n_parallel > 1:
        singleton_pool.run_each(_worker_init_tf)
    if horizon is None:
        horizon = gpr_env.horizon
    if (env, policy) not in _cached_sampler:
        vec_sampler = VectorizedSampler(
            env=env,
            policy=policy,
            n_envs=len(seeds),
        )
        vec_sampler.start_worker()
        _cached_sampler[env, policy] = vec_sampler
    else:
        vec_sampler = _cached_sampler[env, policy]
    assert vec_sampler.n_envs == len(seeds)
    vec_gpr_env = get_vec_gpr_env(vec_sampler.vec_env)
    vec_gpr_env.inject_noise(noise_levels)
    if stagewise is None:
        stagewise = False
    vec_gpr_env.set_stagewise(stagewise)  # set_stagewise(stagewise)
    vec_gpr_env.set_xinits(xinits)
    paths = vec_sampler.obtain_samples(max_path_length=horizon, batch_size=len(seeds) * horizon,
                                       max_n_trajs=len(seeds), seeds=seeds, show_progress=show_progress)
    return paths


def path_to_traj(gpr_env, path):
    xs = path["env_infos"]["x"]
    trajectory = Trajectory(env=gpr_env)
    trajectory.solution = dict(
        xinit=xs[0],
        start_time=0,
        end_time=0,
        env=gpr_env,
        x=xs,
        reward=path["rewards"],
        qpos=xs[:, :gpr_env.world.model.dimq],
        qvel=xs[:, gpr_env.world.model.dimq:],
        u=path["actions"]
    )
    return trajectory


class FetchWrapperPolicy(Policy, Serializable):
    def __init__(self, env_spec, wrapped_policy):
        assert isinstance(wrapped_policy.distribution, DiagonalGaussian)
        Serializable.quick_init(self, locals())
        self.wrapped_policy = wrapped_policy
        Policy.__init__(self, env_spec)
        self._dist = DiagonalGaussian(8)

    def dist_info_sym(self, obs_var, state_info_vars=None):
        dist_info = self.wrapped_policy.dist_info_sym(obs_var)
        N = tf.shape(obs_var)[0]
        mean = dist_info['mean']
        log_std = dist_info['log_std']
        expanded_mean = tf.concat(concat_dim=1, values=[
            tf.slice(mean, [0, 0], [-1, 3]),
            tf.zeros(tf.pack([N, 3])),
            tf.tile(
                tf.slice(mean, [0, 3], [-1, 1]),
                [1, 2]
            )
        ])
        expanded_log_std = tf.concat(concat_dim=1, values=[
            tf.slice(log_std, [0, 0], [-1, 3]),
            tf.ones(tf.pack([N, 3])) * -10,
            tf.tile(
                tf.slice(log_std, [0, 3], [-1, 1]),
                [1, 2]
            )
        ])
        return dict(
            mean=expanded_mean,
            log_std=expanded_log_std
        )

    def get_params_internal(self, **tags):
        return self.wrapped_policy.get_params(**tags)

    def get_actions(self, observations):
        actions, agent_infos = self.wrapped_policy.get_actions(observations)
        N = len(observations)
        expanded_actions = np.concatenate([
            actions[:, [0, 1, 2]],
            np.zeros((N, 3)),
            actions[:, [3, 3]]
        ], axis=1)
        mean = agent_infos["mean"]
        log_std = agent_infos["log_std"]
        expanded_mean = np.concatenate([
            mean[:, [0, 1, 2]],
            np.zeros((N, 3)),
            mean[:, [3, 3]]
        ], axis=1)
        expanded_log_std = np.concatenate([
            log_std[:, [0, 1, 2]],
            np.ones((N, 3)) * -10,
            log_std[:, [3, 3]]
        ], axis=1)
        return expanded_actions, dict(mean=expanded_mean, log_std=expanded_log_std)

    def get_action(self, observation):
        actions, agent_infos = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in agent_infos.items()}

    @property
    def distribution(self):
        return self._dist

    def log_diagnostics(self, paths):
        self.wrapped_policy.log_diagnostics(paths)

    @property
    def vectorized(self):
        return self.wrapped_policy.vectorized

    def reset(self, dones=None):
        self.wrapped_policy.reset(dones=dones)


class DiscretizedFetchWrapperPolicy(Policy, Serializable):
    def __init__(self, wrapped_policy, disc_intervals, expectation=None):
        Serializable.quick_init(self, locals())
        Policy.__init__(self, wrapped_policy.env_spec)
        self.wrapped_policy = wrapped_policy
        self.disc_intervals = list(map(np.asarray, disc_intervals))
        self.expectation = expectation

    def get_params_internal(self, **tags):
        return self.wrapped_policy.get_params_internal(**tags)

    def reset(self, dones=None):
        self.wrapped_policy.reset(dones)

    def get_actions(self, observations):
        actions, agent_infos = self.wrapped_policy.get_actions(observations)
        actions = np.asarray(actions)
        if self.expectation == 'mean':
            numeric_actions = []
            for idx in range(3):
                numeric_actions.append(
                    np.sum(
                        np.asarray(self.disc_intervals[idx]) * np.asarray(agent_infos['id_{}_prob'.format(idx)]),
                        axis=-1
                    )
                )
            numeric_actions.extend(np.zeros((3, len(observations))))
            numeric_actions.append(np.sum(np.asarray([-1, 1]) * np.asarray(agent_infos['id_3_prob']), -1))
            numeric_actions.append(np.sum(np.asarray([-1, 1]) * np.asarray(agent_infos['id_3_prob']), -1))
        if self.expectation == 'log_mean':
            numeric_actions = []
            for idx in range(3):
                import ipdb;
                ipdb.set_trace()
                numeric_actions.append(
                    np.sum(
                        np.log(self.disc_intervals[idx]) * np.asarray(agent_infos['id_{}_prob'.format(idx)]),
                        axis=-1
                    )
                )
            numeric_actions.extend(np.zeros((3, len(observations))))
            numeric_actions.append(np.sum(np.asarray([-1, 1]) * np.asarray(agent_infos['id_3_prob']), -1))
            numeric_actions.append(np.sum(np.asarray([-1, 1]) * np.asarray(agent_infos['id_3_prob']), -1))
        elif self.expectation is None:
            numeric_actions = []
            for idx in range(3):
                numeric_actions.append(self.disc_intervals[idx][actions[:, idx]])
            numeric_actions.extend(np.zeros((3, len(observations))))
            numeric_actions.append(np.asarray([-1, 1])[actions[:, -1]])
            numeric_actions.append(np.asarray([-1, 1])[actions[:, -1]])
        else:
            raise NotImplementedError
        return np.asarray(numeric_actions).T, agent_infos

    def get_action(self, observation):
        actions, agent_infos = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in agent_infos.items()}

    def get_greedy_actions(self, observations):
        _, agent_infos = self.wrapped_policy.get_actions(observations)
        actions = np.asarray(self.distribution.maximum_a_posteriori(agent_infos))
        numeric_actions = []
        for idx in range(3):
            numeric_actions.append(self.disc_intervals[idx][actions[:, idx]])
        numeric_actions.extend(np.zeros((3, len(observations))))
        numeric_actions.append(np.asarray([-1, 1])[actions[:, -1]])
        numeric_actions.append(np.asarray([-1, 1])[actions[:, -1]])
        return np.asarray(numeric_actions).T, agent_infos

        # _, infos = self.get_actions()

    @property
    def distribution(self):
        return self.wrapped_policy.distribution


def discretized_env_spec(spec, disc_intervals):
    components = []
    for bins in disc_intervals:
        components.append(Discrete(len(bins)))
    # action for the gripper
    components.append(Discrete(2))
    return EnvSpec(
        observation_space=spec.observation_space,
        action_space=Product(components),
    )


class CircularQueue(object):
    def __init__(self, max_size, data_shape):
        self.data = np.empty((max_size,) + data_shape)
        self.size = 0
        self.max_size = max_size
        self.head = 0

    def extend(self, samples):
        N = len(samples)
        # if there's more samples than the max size of the pool, directly fill it (unlikely in our case)
        if N >= self.max_size:
            self.data[:] = samples[-self.max_size:]
            self.head = 0
            self.size = self.max_size
        elif self.head + N <= self.max_size:
            self.data[self.head:self.head + N] = samples
            self.head += N
        else:
            self.data[self.head:self.max_size] = samples[:self.max_size - self.head]
            self.data[:N - (self.max_size - self.head)] = samples[self.max_size - self.head:]
            self.head = N - (self.max_size - self.head)
        self.size = min(self.size + N, self.max_size)

    def iterate(self, n_batches, batch_size):
        for _ in range(n_batches):
            ids = np.random.randint(0, high=self.size, size=batch_size)
            yield self.data[ids]

    def clear(self):
        self.size = 0
        self.head = 0


class MixturePolicy(object):
    def __init__(self, policies, ratios):
        self.policies = policies
        self.ratios = np.asarray(ratios)

    @property
    def vectorized(self):
        return True

    def reset(self, dones=None):
        for policy in self.policies:
            policy.reset(dones=dones)

    def inform_vec_env(self, vec_env):
        for policy in self.policies:
            if hasattr(policy, 'inform_vec_env'):
                policy.inform_vec_env(vec_env)

    def get_actions(self, observations):
        N = len(observations)
        policy_actions = []
        for policy in self.policies:
            policy_actions.append(policy.get_actions(observations)[0])
        assignments = weighted_sample_n(
            np.tile(
                self.ratios.reshape((1, -1)),
                (N, 1)
            ),
            np.arange(len(self.policies))
        )
        return np.asarray(policy_actions)[assignments, np.arange(N)], dict()


def clipped_square_loss(x, threshold):
    return tf.select(tf.abs(x) < threshold, 0.5 * tf.square(x), threshold * (tf.abs(x) - 0.5 * threshold))


def bc_trainer(env, policy, max_n_samples=None, clip_square_loss=None):
    obs_var = env.observation_space.new_tensor_variable(extra_dims=1, name="obs")
    obs_dim = env.observation_space.flat_dim

    if isinstance(policy, DiscretizedFetchWrapperPolicy):  # .distribution, DiagonalGaussian):
        action_var = policy.wrapped_policy.action_space.new_tensor_variable(extra_dims=1, name="action")
        action_dim = policy.wrapped_policy.action_space.flat_dim
        pol_dist_info_vars = policy.wrapped_policy.dist_info_sym(obs_var)
        logli_var = policy.wrapped_policy.distribution.log_likelihood_sym(action_var, pol_dist_info_vars)
        loss_var = -tf.reduce_mean(logli_var)
    else:
        action_var = env.action_space.new_tensor_variable(extra_dims=1, name="action")
        action_dim = env.action_space.flat_dim
        pol_action_var = policy.dist_info_sym(obs_var)["mean"]

        if clip_square_loss is not None:
            loss_var = tf.reduce_mean(clipped_square_loss(action_var - pol_action_var, clip_square_loss))
        else:
            loss_var = tf.reduce_mean(0.5 * tf.square(action_var - pol_action_var))

    train_op = tf.train.AdamOptimizer().minimize(loss_var, var_list=policy.get_params())

    data_pool = CircularQueue(max_size=max_n_samples, data_shape=(obs_dim + action_dim,))

    def add_paths(new_paths):
        if isinstance(policy, DiscretizedFetchWrapperPolicy):
            intervals = policy.disc_intervals

            for p in new_paths:
                disc_actions = []
                for disc_idx in range(3):
                    cur_actions = p["actions"][:, disc_idx]
                    bins = np.asarray(intervals[disc_idx])
                    disc_actions.append(np.argmin(np.abs(cur_actions[:, None] - bins[None, :]), axis=1))
                disc_actions.append(np.cast['uint8'](p["actions"][:, -1] == 1))  # p["actions"][:, -1])
                flat_actions = policy.wrapped_policy.action_space.flatten_n(np.asarray(disc_actions).T)
                data_pool.extend(np.concatenate([p["observations"], flat_actions], axis=1))
        else:
            for p in new_paths:
                data_pool.extend(np.concatenate([p["observations"], p["actions"]], axis=1))

    def train_loop(batch_size, n_updates_per_epoch):

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for epoch_idx in itertools.count():
                logger.log("Starting epoch {}".format(epoch_idx))
                loss_vals = []
                logger.log("Start training")
                progbar = pyprind.ProgBar(n_updates_per_epoch)

                for batch in data_pool.iterate(n_batches=n_updates_per_epoch, batch_size=batch_size):
                    batch_obs = batch[..., :obs_dim]
                    batch_actions = batch[..., obs_dim:]
                    _, loss_val = sess.run(
                        [train_op, loss_var],
                        feed_dict={obs_var: batch_obs, action_var: batch_actions}
                    )
                    loss_vals.append(loss_val)
                    progbar.update()

                logger.log("Finished training")
                if progbar.active:
                    progbar.stop()
                logger.record_tabular('Epoch', epoch_idx)
                logger.record_tabular('Loss', np.mean(loss_vals))
                logger.record_tabular('NSamples', data_pool.size)
                logger.save_itr_params(epoch_idx, params=dict(
                    env=env, policy=policy
                ))
                yield epoch_idx

    return AttrDict(
        add_paths=add_paths,
        train_loop=train_loop,
    )


class DeterministicPolicy(Policy, Serializable):
    def __init__(self, env_spec, wrapped_policy):
        Serializable.quick_init(self, locals())
        self.wrapped_policy = wrapped_policy
        Policy.__init__(self, env_spec)

    def get_params_internal(self, **tags):
        return self.wrapped_policy.get_params(**tags)

    def get_actions(self, observations):
        if hasattr(self.wrapped_policy, "get_greedy_actions"):
            return self.wrapped_policy.get_greedy_actions(observations)
        _, agent_infos = self.wrapped_policy.get_actions(observations)
        return self.wrapped_policy.distribution.maximum_a_posteriori(agent_infos), dict()

    def get_action(self, observation):
        actions, agent_infos = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in agent_infos.items()}

    def reset(self, dones=None):
        self.wrapped_policy.reset(dones=dones)

    @property
    def vectorized(self):
        return self.wrapped_policy.vectorized


def find_stageinit_points(env, path):
    site_xpos = path["env_infos"]["site_xpos"]
    # assume that there are two additional sites, stall_mocap and grip
    n_geoms = (site_xpos.shape[1] - 6) // 3
    pts = []
    for stage in range(n_geoms):
        candidates = np.where(path["env_infos"]["stage"] == stage)[0]
        if len(pts) > 0:
            candidates = [x for x in candidates if x >= pts[-1]]
        if len(candidates) > 0:
            pts.append(candidates[0])
        else:
            break
    return pts


def compute_stage(env, site_xpos):
    """
    Compute the stage separately for a list of site positions
    """
    n_geoms = (site_xpos.shape[1] - 6) // 3
    geom_xpos = site_xpos[:, 6:].reshape((-1, n_geoms, 3)).transpose((1, 0, 2))
    gpr_env = get_gpr_env(env)
    block_order = gpr_env.task_id[0]
    geom_xpos = geom_xpos[block_order]
    completed = np.empty((len(geom_xpos) - 1, len(site_xpos)), dtype=np.bool)
    for idx, (xpos0, xpos1) in enumerate(zip(geom_xpos, geom_xpos[1:])):
        completed[idx, :] = np.logical_and(
            np.logical_and(
                np.abs(xpos1[:, 0] - xpos0[:, 0]) < 0.02,
                np.abs(xpos1[:, 1] - xpos0[:, 1]) < 0.02
            ),
            np.abs(xpos1[:, 2] - 0.05 - xpos0[:, 2]) < 0.005
        )
    stages = np.cumprod(completed, axis=0).sum(axis=0)
    return stages


class DiscretizedEnvWrapper(ProxyEnv, Serializable):
    def __init__(self, wrapped_env, disc_intervals):
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, wrapped_env)
        self.disc_intervals = disc_intervals
        components = []
        for bins in disc_intervals:
            components.append(Discrete(len(bins)))
        # action for the gripper
        components.append(Discrete(2))
        self._action_space = Product(components)

    @property
    def action_space(self):
        return self._action_space

    def step(self, action):
        numeric_action = []
        for idx in range(3):
            numeric_action.append(self.disc_intervals[idx][action[idx]])
        numeric_action.extend([0, 0, 0])
        numeric_action.append([-1, 1][action[-1]])
        numeric_action.append([-1, 1][action[-1]])
        return self.wrapped_env.step(np.asarray(numeric_action))

    @property
    def vectorized(self):
        return getattr(self.wrapped_env, "vectorized", False)

    def vec_env_executor(self, n_envs):
        return VecDiscretizedEnv(self, n_envs)


class VecDiscretizedEnv(VecEnv):
    def __init__(self, env: DiscretizedEnvWrapper, n_envs):
        self.env = env
        self.vec_env = env.wrapped_env.vec_env_executor(n_envs)
        self.n_envs = n_envs

    def reset(self, dones, seeds=None, *args, **kwargs):
        return self.vec_env.reset(dones=dones, seeds=seeds, *args, **kwargs)

    def step(self, action_n, max_path_length=None):
        action_n = np.asarray(action_n)
        numeric_actions = []
        for idx in range(3):
            numeric_actions.append(self.env.disc_intervals[idx][action_n[:, idx]])
        numeric_actions.extend(np.zeros((3, len(action_n))))
        numeric_actions.append(np.asarray([-1, 1])[action_n[:, -1]])
        numeric_actions.append(np.asarray([-1, 1])[action_n[:, -1]])
        numeric_actions = np.asarray(numeric_actions).T
        return self.vec_env.step(numeric_actions, max_path_length)


custom_py_cnt = 0


class SoftmaxExactEntropy(Parameterized, Serializable):
    """
    Directly parametrize the entropy of the softmax function.

    TODO
    """

    def __init__(self, dim, input_dependent=False, initial_entropy_percentage=0.99, bias=3.0):
        Serializable.quick_init(self, locals())
        Parameterized.__init__(self)
        self.dim = dim
        self.max_ent = np.log(dim)
        self.input_dependent = input_dependent
        self.bias = bias
        if input_dependent:
            self.p = None
        else:
            # we parametrize the entropy as max_ent * sigmoid(lambda)
            self.p = tf.Variable(initial_value=logit(initial_entropy_percentage), name="p", dtype=tf.float32)

    @property
    def flat_dim(self):
        if self.input_dependent:
            return self.dim + 1
        else:
            return self.dim

    def activate(self, x):
        global custom_py_cnt
        custom_py_cnt += 1

        if self.input_dependent:
            desired_ent = self.max_ent * tf.nn.sigmoid(x[:, self.dim] + self.bias)
            desired_ent = 0.01 + 0.98 * desired_ent
            x = x[:, :self.dim]
        else:
            desired_ent = self.max_ent * tf.nn.sigmoid(self.p)
            desired_ent = 0.01 + 0.98 * desired_ent

            desired_ent = tf.tile(tf.pack([desired_ent]), tf.pack([tf.shape(x)[0]]))

        func_name = "CustomPyFunc%d" % custom_py_cnt

        @tf.RegisterGradient(func_name)
        def _grad(op, grad):
            dx, dh = tf.py_func(exact_softmax_t_grad, [op.inputs[0], op.inputs[1], grad], [op.inputs[0].dtype,
                                                                                           op.inputs[1].dtype])
            dx.set_shape(op.inputs[0].get_shape())
            dh.set_shape(op.inputs[1].get_shape())
            return dx, dh

        @tf.RegisterShape(func_name)
        def _shape(op):
            return op.inputs[0].get_shape()

        g = tf.get_default_graph()
        with g.gradient_override_map({"PyFunc": func_name}):
            ts, = tf.py_func(exact_softmax_t, [x, desired_ent], [x.dtype])
            # ts.set_shape(x.get_shape()[:-1])
            ret = tf.nn.softmax(x / tf.expand_dims(ts, -1))
            ret.set_shape(x.get_shape())
            return ret

    def get_params_internal(self, **tags):
        if not self.input_dependent and tags.get('trainable', True):
            return [self.p]
        return []


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    x = np.exp(x)
    x = x / np.sum(x, axis=-1, keepdims=True)
    return x


def ent(x):
    return np.sum(-x * np.log(x + 1e-8), axis=-1)


def exact_softmax_op(x, desired_ents):
    global custom_py_cnt
    custom_py_cnt += 1
    func_name = "CustomPyFunc%d" % custom_py_cnt

    @tf.RegisterGradient(func_name)
    def _grad(op, grad):
        dx, dh = tf.py_func(exact_softmax_t_grad, [op.inputs[0], op.inputs[1], grad], [op.inputs[0].dtype,
                                                                                       op.inputs[1].dtype])
        dx.set_shape(op.inputs[0].get_shape())
        dh.set_shape(op.inputs[1].get_shape())
        return dx, dh

    @tf.RegisterShape(func_name)
    def _shape(op):
        return op.inputs[0].get_shape()

    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": func_name}):
        ts, = tf.py_func(exact_softmax_t, [x, desired_ents], [x.dtype])
        ret = tf.nn.softmax(x / tf.expand_dims(ts, -1))
        ret.set_shape(x.get_shape())
        return ret


def exact_softmax_t(x, desired_ents):
    # solve for t (separate t for each componetn) such that ent(softmax(x/t)) ~ desired_ent

    low_ts = np.zeros(x.shape[:-1])
    high_ts = np.ones(x.shape[:-1])

    # increase high_ts until lowest ent is greater than desired_ent

    n_itrs = 0

    for _ in range(10):
        # while True:
        n_itrs += 1
        cur_ent = ent(softmax(x / high_ts[:, np.newaxis]))
        if np.any(cur_ent < desired_ents):
            high_ts *= 2
        else:
            break

    for _ in range(40):

        # while True:
        n_itrs += 1
        ts = (low_ts + high_ts) / 2
        cur_ent = ent(softmax(x / ts[:, np.newaxis]))
        # if cur_ent < desired_ent, should decrease temp to make it more random
        high_mask = cur_ent > desired_ents
        low_mask = cur_ent <= desired_ents
        high_ts[high_mask] = ts[high_mask]
        low_ts[low_mask] = ts[low_mask]
        if np.max(np.abs(cur_ent - desired_ents)) < 1e-6:
            break

    return np.cast['float32'](ts)


def exact_softmax_t_grad(x, desired_ents, grad_output):
    # Implementation of the gradient has been checked against sympy, and should be correct
    ts = np.expand_dims(exact_softmax_t(x, desired_ents), -1)

    xts = x / ts
    # same numerical stability trick as softmax
    expxts = np.exp(xts - np.max(xts, axis=-1, keepdims=True))
    sumexpxts = np.sum(expxts, axis=-1, keepdims=True)
    sumxexpxts = np.sum(x * expxts, axis=-1, keepdims=True)
    sumx2expxts = np.sum(np.square(x) * expxts, axis=-1, keepdims=True)

    # compute grad w.r.t. x
    numerator = ts * (x * sumexpxts - sumxexpxts) * expxts
    denominator = sumx2expxts * sumexpxts - sumxexpxts ** 2

    x_grads = np.cast['float32'](numerator / (denominator + 1e-8))

    ent_grads = np.cast['float32'](- ts ** 3 * (sumexpxts ** 2) / (sumxexpxts ** 2 - sumx2expxts * sumexpxts))
    ent_grads = ent_grads[:, 0]

    return x_grads * grad_output[:, np.newaxis], ent_grads * grad_output


def sigmoid(x): return 1. / (1 + np.exp(-x))


def logit(x): return np.log(x / (1 - x))


class EntropyControlledPolicy(Policy, Serializable):
    def __init__(self, wrapped_policy: Policy, initial_entropy):
        Serializable.quick_init(self, locals())
        Policy.__init__(self, wrapped_policy.env_spec)
        self.wrapped_policy = wrapped_policy
        # parameterize entropy by max_ent * tf.sigmoid(p)
        max_ents = np.log([x.n for x in self.wrapped_policy.action_space.components])

        initial_entropy = np.asarray(initial_entropy)
        # solve for the initiaal ent param
        initial_ent_param = logit((np.maximum(0.0011, initial_entropy) - 0.001) / 0.998 / max_ents)

        # assert np.allclose(initial_entropy, max_ents * sigmoid(initial_ent_param) * 0.998 + 0.001)

        self.ent_param = tf.Variable(initial_value=initial_ent_param, dtype=tf.float32, trainable=True, name="ent")
        desired_ent = max_ents * tf.sigmoid(self.ent_param)
        # lower and upper bound the entropy
        desired_ent = 0.001 + 0.998 * desired_ent
        self.desired_ent = desired_ent

        obs_var = self.wrapped_policy.env_spec.observation_space.new_tensor_variable(extra_dims=1, name="obs")
        self._f_dist_info = tensor_utils.compile_function(
            [obs_var],
            self.dist_info_sym(obs_var)
        )

    def get_params_internal(self, **tags):
        params = self.wrapped_policy.get_params_internal(**tags)
        if tags.get('trainable', True):
            params = params + [self.ent_param]
        return params

    @property
    def distribution(self):
        return self.wrapped_policy.distribution

    def dist_info_sym(self, obs_var, state_info_vars=None):
        dist_infos = self.wrapped_policy.dist_info_sym(obs_var)
        for idx, (k, v) in enumerate(sorted(list(dist_infos.items()))):
            logits = tf.log(dist_infos[k])
            desired_ent = tf.tile(tf.pack([self.desired_ent[idx]]), tf.pack([tf.shape(obs_var)[0]]))
            dist_infos[k] = exact_softmax_op(logits, desired_ent)  # self.desired_ent[idx])
        return dist_infos

    def get_action(self, observation):
        actions, agent_infos = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in agent_infos.items()}

    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        dist_info = self._f_dist_info(flat_obs)
        return np.asarray(self.distribution.sample(dist_info)), dist_info

    @property
    def vectorized(self):
        return self.wrapped_policy.vectorized


def worker_collect_xinit(G, height, seed):
    env = gpr_fetch_env(horizon=1, height=height)
    env.seed(int(seed))
    env.reset()
    return env.x


def collect_xinits(height, seeds):
    xs = []
    pbar = pyprind.ProgBar(len(seeds))
    for x in singleton_pool.run_imap_unordered(worker_collect_xinit, [(height, seed) for seed in seeds]):
        xs.append(x)
        pbar.update()
    if pbar.active:
        pbar.stop()
    return np.asarray(xs)


def analogy_bc_trainer(env_spec, policy, max_n_samples=None, clip_square_loss=None):
    obs_var = env_spec.observation_space.new_tensor_variable(extra_dims=1, name="obs")
    obs_dim = env_spec.observation_space.flat_dim

    if isinstance(policy, DiscretizedFetchWrapperPolicy):
        action_var = policy.wrapped_policy.action_space.new_tensor_variable(extra_dims=1, name="action")
        action_dim = policy.wrapped_policy.action_space.flat_dim
        pol_dist_info_vars = policy.wrapped_policy.dist_info_sym(obs_var)
        logli_var = policy.wrapped_policy.distribution.log_likelihood_sym(action_var, pol_dist_info_vars)
        loss_var = -tf.reduce_mean(logli_var)
    else:
        action_var = env_spec.action_space.new_tensor_variable(extra_dims=1, name="action")
        action_dim = env_spec.action_space.flat_dim
        pol_action_var = policy.dist_info_sym(obs_var)["mean"]

        if clip_square_loss is not None:
            loss_var = tf.reduce_mean(clipped_square_loss(action_var - pol_action_var, clip_square_loss))
        else:
            loss_var = tf.reduce_mean(0.5 * tf.square(action_var - pol_action_var))

    train_op = tf.train.AdamOptimizer().minimize(loss_var, var_list=policy.get_params())

    # traj_pool =

    data_pool = CircularQueue(max_size=max_n_samples, data_shape=(obs_dim + action_dim,))

    def add_paths(new_paths):
        if isinstance(policy, DiscretizedFetchWrapperPolicy):
            intervals = policy.disc_intervals

            for p in new_paths:
                disc_actions = []
                for disc_idx in range(3):
                    cur_actions = p["actions"][:, disc_idx]
                    bins = np.asarray(intervals[disc_idx])
                    disc_actions.append(np.argmin(np.abs(cur_actions[:, None] - bins[None, :]), axis=1))
                disc_actions.append(np.cast['uint8'](p["actions"][:, -1] == 1))  # p["actions"][:, -1])
                flat_actions = policy.wrapped_policy.action_space.flatten_n(np.asarray(disc_actions).T)
                data_pool.extend(np.concatenate([p["observations"], flat_actions], axis=1))
        else:
            for p in new_paths:
                data_pool.extend(np.concatenate([p["observations"], p["actions"]], axis=1))

    def train_loop(batch_size, n_updates_per_epoch):

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for epoch_idx in itertools.count():
                logger.log("Starting epoch {}".format(epoch_idx))
                loss_vals = []
                logger.log("Start training")
                progbar = pyprind.ProgBar(n_updates_per_epoch)

                for batch in data_pool.iterate(n_batches=n_updates_per_epoch, batch_size=batch_size):
                    batch_obs = batch[..., :obs_dim]
                    batch_actions = batch[..., obs_dim:]
                    _, loss_val = sess.run(
                        [train_op, loss_var],
                        feed_dict={obs_var: batch_obs, action_var: batch_actions}
                    )
                    loss_vals.append(loss_val)
                    progbar.update()

                logger.log("Finished training")
                if progbar.active:
                    progbar.stop()
                logger.record_tabular('Epoch', epoch_idx)
                logger.record_tabular('Loss', np.mean(loss_vals))
                logger.record_tabular('NSamples', data_pool.size)
                logger.save_itr_params(epoch_idx, params=dict(
                    env=env, policy=policy
                ))
                yield epoch_idx

    return AttrDict(
        add_paths=add_paths,
        train_loop=train_loop,
    )
