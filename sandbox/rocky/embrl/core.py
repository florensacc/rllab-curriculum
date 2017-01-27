import tensorflow as tf
from rllab.envs.base import Env
from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.sampler.utils import rollout
from sandbox.rocky.tf.algos.trpo import TRPO
import numpy as np

from sandbox.rocky.tf.envs.vec_env import VecEnv


class EnsembleModelEnv(Env, Serializable):
    def __init__(self, env_spec, models, model_output_space):
        Serializable.quick_init(self, locals())
        self._observation_space = env_spec.observation_space
        self._action_space = env_spec.action_space
        self.model_output_space = model_output_space
        self.models = models
        self.initial_states = []
        self.vec_env = self.vec_env_executor(n_envs=1)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def reset(self):
        return self.vec_env.reset(dones=[True])[0]

    def step(self, action):
        next_obs, rewards, dones, infos = self.vec_env.step(action_n=[action], max_path_length=None)
        return next_obs[0], rewards[0], dones[0], {k: v[0] for k, v in infos.items()}

    @property
    def vectorized(self):
        return True

    def vec_env_executor(self, n_envs):
        return VecEnsembleModelEnv(self, n_envs)

    def fit(self, paths):
        logger.log("Processing data")
        obs = np.concatenate([p["observations"] for p in paths], axis=0)
        actions = np.concatenate([p["actions"] for p in paths], axis=0)
        next_obs = []
        for path in paths:
            next_obs.extend(self.observation_space.unflatten_n(
                list(path["observations"][1:]) + [path["last_obs"]]
            ))
        rewards = np.concatenate([p["rewards"] for p in paths], axis=0)
        dones = np.concatenate([p["dones"] for p in paths], axis=0)
        targets = self.model_output_space.flatten_n(list(zip(next_obs, rewards, dones)))
        inputs = np.concatenate([obs, actions], axis=1)
        self.initial_states = np.asarray([p["observations"][0] for p in paths])
        for idx, model in enumerate(self.models):
            logger.log("Fitting model {}".format(idx))
            model.fit(inputs, targets)


class VecEnsembleModelEnv(VecEnv):
    def __init__(self, env: EnsembleModelEnv, n_envs):
        self.env = env
        self.n_envs = n_envs
        self.ts = np.zeros(n_envs, dtype=np.int)
        self.model_ids = np.zeros(n_envs, dtype=np.int)
        self.states = np.zeros((n_envs, env.observation_space.flat_dim))

    def reset(self, dones, seeds=None, *args, **kwargs):
        assert seeds is None
        dones = np.cast['bool'](dones)
        n_dones = np.sum(dones)
        if n_dones == 0:
            return []
        self.ts[dones] = 0
        self.model_ids[dones] = np.random.randint(low=0, high=len(self.env.models), size=n_dones)
        state_ids = np.random.choice(np.arange(len(self.env.initial_states)), size=n_dones, replace=True)
        self.states[dones] = self.env.initial_states[state_ids]
        return self.env.observation_space.unflatten_n(self.states[dones])

    def step(self, action_n, max_path_length=None):
        action_n = self.env.action_space.flatten_n(action_n)
        rewards = np.zeros(self.n_envs)
        dones = np.zeros(self.n_envs, dtype=np.bool)
        self.ts += 1
        for model_id, model in enumerate(self.env.models):
            mask = self.model_ids == model_id
            if np.any(mask):
                preds = model.predict_sample(np.concatenate([self.states[mask], action_n[mask]], axis=1))
                self.states[mask], rewards[mask], dones[mask] = list(map(np.asarray, zip(*preds)))
        rewards = np.clip(rewards, -1, 1)
        self.states = np.clip(self.states, -10, 10)
        if max_path_length is not None:
            dones[self.ts >= max_path_length] = True
        return np.copy(self.states), rewards, dones, dict()


class EMBRL(object):
    def __init__(self, env, policy, ensemble_env, n_itr, trpo_args):
        self.env = env
        self.policy = policy
        self.ensemble_env = ensemble_env
        self.n_itr = n_itr
        self.trpo_args = trpo_args

    def train(self):
        with tf.Session() as sess:
            logger.log("Initializing variables")
            sess.run(tf.initialize_all_variables())
            logger.log("Initialized")

            init_params = self.policy.get_param_values()

            paths = []

            for _ in range(self.n_itr):
                logger.log("Sampling in real environment")
                path = rollout(self.env, self.policy, max_path_length=self.env.horizon)
                logger.record_tabular('RealReturn', np.sum(path["rewards"]))
                logger.dump_tabular()
                paths.append(path)

                self.ensemble_env.fit(paths)

                self.policy.set_param_values(init_params)
                trpo = TRPO(
                    env=self.ensemble_env,
                    policy=self.policy,
                    **self.trpo_args,
                )
                trpo.train(sess=sess)
