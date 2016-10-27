# Synchronous Advantage Actor-Critic
import pickle

import pyprind
import tensorflow as tf
from cached_property import cached_property

import sandbox.rocky.tf.core.layers as L
from sandbox.rocky.neural_learner.samplers.vectorized_sampler import VectorizedSampler
from sandbox.rocky.tf.core.layers_powered import LayersPowered
import numpy as np

from sandbox.rocky.tf.core.parameterized import suppress_params_loading
from sandbox.rocky.tf.envs.vec_env_executor import VecEnvExecutor
from sandbox.rocky.tf.policies.base import StochasticPolicy
from sandbox.rocky.tf.misc import tensor_utils
from rllab.core.serializable import Serializable
from sandbox.rocky.tf.distributions.categorical import Categorical
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.misc import logger
from rllab.misc import special
from contextlib import contextmanager
import time


class PolicyWrapper(StochasticPolicy, LayersPowered, Serializable):
    def __init__(self, env_spec, ac):
        Serializable.quick_init(self, locals())
        super(PolicyWrapper, self).__init__(env_spec)
        self.ac = ac
        self.f_dist = tensor_utils.compile_function(
            inputs=[self.ac.l_in.input_var],
            outputs=self.dist_info_sym(self.ac.l_in.input_var, dict())
        )
        self.f_joint_dist_vf = tensor_utils.compile_function(
            inputs=[self.ac.l_in.input_var],
            outputs=self.ac.joint_dist_info_vf(self.ac.l_in.input_var, dict()),
        )
        LayersPowered.__init__(self, [self.ac.l_prob])

    @cached_property
    def distribution(self):
        return Categorical(self.action_space.flat_dim)

    def dist_info_sym(self, obs_var, state_info_vars):
        prob = L.get_output(self.ac.l_prob, {self.ac.l_in: obs_var})
        return dict(prob=prob)

    @property
    def vectorized(self):
        return True

    def get_action(self, observation):
        actions, agent_infos = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in agent_infos.items()}

    def get_actions(self, observations):
        dist_info = self.f_joint_dist_vf(observations)
        prob = dist_info["prob"]
        actions = special.weighted_sample_n(prob, np.arange(self.action_space.flat_dim))
        return actions, dist_info


class VfWrapper(LayersPowered, Serializable):
    def __init__(self, env_spec, ac):
        Serializable.quick_init(self, locals())
        self.ac = ac
        LayersPowered.__init__(self, [self.ac.l_v])
        self.f_predict = tensor_utils.compile_function(
            inputs=[self.ac.l_in.input_var],
            outputs=tf.reshape(L.get_output(self.ac.l_v), (-1,)),
        )
        self.env_spec = env_spec
        self.observation_space = env_spec.observation_space

    def predict(self, observations):
        return self.f_predict(observations)

    def predict_sym(self, obs_var):
        return tf.reshape(L.get_output(self.ac.l_v, {self.ac.l_in: obs_var}), (-1,))


class JointActorCritic(LayersPowered, Serializable):
    def __init__(self, env_spec, hidden_sizes=(32, 32)):
        Serializable.quick_init(self, locals())
        with tf.variable_scope("ram_head"):
            obs_dim = env_spec.observation_space.flat_dim
            action_dim = env_spec.action_space.flat_dim
            l_in = L.InputLayer(
                shape=(None, obs_dim),
                name="input"
            )
            l_hid = l_in
            for hid_size in hidden_sizes:
                l_hid = L.DenseLayer(
                    l_in,
                    num_units=hid_size,
                    nonlinearity=tf.nn.relu,
                )
            l_feature = l_hid

            l_prob = L.DenseLayer(
                l_feature,
                num_units=action_dim,
                name="prob",
                nonlinearity=tf.nn.softmax,
            )

            l_v = L.DenseLayer(
                l_feature,
                num_units=1,
                name="v",
            )

            self.l_in = l_in
            self.l_feature = l_feature
            self.l_prob = l_prob
            self.l_v = l_v
            self.env_spec = env_spec

            LayersPowered.__init__(self, [l_prob, l_v])
            self.policy = PolicyWrapper(self.env_spec, self)
            self.vf = VfWrapper(self.env_spec, self)

    def joint_dist_info_vf(self, obs_var, state_info_vars):
        prob, vs = L.get_output([self.l_prob, self.l_v], {self.l_in: obs_var})
        vs = tf.reshape(vs, (-1,))
        return dict(prob=prob, _vs=vs)


class ImageJointActorCritic(LayersPowered, Serializable):
    def __init__(self, env_spec, hidden_nonlinearity=tf.nn.relu, W=None, b=None):
        Serializable.quick_init(self, locals())
        with tf.variable_scope("dqn_head"):
            if W is None:
                W = L.HeUniformInitializer()
            if b is None:
                b = tf.constant_initializer(0.1)
            obs_dim = env_spec.observation_space.flat_dim
            action_dim = env_spec.action_space.flat_dim
            l_in = L.InputLayer(
                shape=(None,) + env_spec.observation_space.shape,
                name="input"
            )
            l_in_reshaped = l_in
            l_hid = L.Conv2DLayer(
                l_in_reshaped, num_filters=16, filter_size=8, stride=4,
                name="conv1", nonlinearity=hidden_nonlinearity, W=W, b=b,
            )
            l_hid = L.Conv2DLayer(
                l_hid, num_filters=32, filter_size=4, stride=2,
                name="conv2", nonlinearity=hidden_nonlinearity, W=W, b=b,
            )
            l_hid = L.DenseLayer(
                l_hid, num_units=env_spec.action_space.flat_dim,
                name="fc", nonlinearity=hidden_nonlinearity, W=W, b=b,
            )

            l_feature = l_hid

            l_prob = L.DenseLayer(
                l_feature,
                num_units=action_dim,
                name="prob",
                nonlinearity=tf.nn.softmax,
                W=W
            )

            l_v = L.DenseLayer(
                l_feature,
                num_units=1,
                name="v",
                W=W,
            )

            self.l_in = l_in
            self.l_feature = l_feature
            self.l_prob = l_prob
            self.l_v = l_v
            self.env_spec = env_spec

            LayersPowered.__init__(self, [l_prob, l_v])
            self.policy = PolicyWrapper(self.env_spec, self)
            self.vf = VfWrapper(self.env_spec, self)

    def joint_dist_info_vf(self, obs_var, state_info_vars):
        prob, vs = L.get_output([self.l_prob, self.l_v], {self.l_in: obs_var})
        vs = tf.reshape(vs, (-1,))
        return dict(prob=prob, _vs=vs)


class TimeLogger(object):
    def __init__(self):
        self.stats = {}

    @contextmanager
    def time(self, key):
        start_time = time.time()
        yield
        end_time = time.time()
        duration = end_time - start_time
        if key not in self.stats:
            self.stats[key] = []
        self.stats[key].append(duration)

    def dump_log(self):
        for k, vs in self.stats.items():
            logger.record_tabular(k, np.sum(vs))

    def reset(self):
        self.stats = {}


class SAAC(Serializable):
    def __init__(self,
                 env,
                 policy,
                 vf,
                 n_envs=8,
                 t_max=5,
                 discount=0.99,
                 policy_loss_coeff=1.,
                 vf_loss_coeff=0.5,
                 entropy_coeff=0.01,
                 n_epochs=1000,
                 epoch_length=1000,
                 # max_steps=8 * 10 ** 7,
                 learning_rate=7e-4,
                 clip_rewards=None,
                 # kl_adjustment_interval=2500,
                 # desired_kl=0.002,
                 post_evals=None,
                 ):
        """
        :type env: Env
        :type policy: StochasticPolicy
        """
        Serializable.quick_init(self, locals())
        self.env = env
        self.policy = policy
        self.vf = vf
        self.n_envs = n_envs
        self.t_max = t_max
        self.discount = discount
        self.policy_loss_coeff = policy_loss_coeff
        self.vf_loss_coeff = vf_loss_coeff
        self.entropy_coeff = entropy_coeff
        self.n_epochs = n_epochs
        self.epoch_length = epoch_length
        self.max_steps = n_epochs * epoch_length
        self.learning_rate = learning_rate
        self.clip_rewards = clip_rewards
        # self.kl_adjustment_interval = kl_adjustment_interval
        # self.desired_kl = desired_kl
        with tf.variable_scope("target_policy"):
            with suppress_params_loading():
                self.target_policy = pickle.loads(pickle.dumps(self.policy))

        if post_evals is None:
            post_evals = []

        for post_eval in post_evals:
            if "env" not in post_eval:
                post_eval["env"] = env
            if "policy" not in post_eval:
                post_eval["policy"] = policy
            if "n_envs" not in post_eval:
                post_eval["n_envs"] = n_envs
            if "sampler" not in post_eval:
                assert post_eval["policy"].vectorized
                post_eval["sampler"] = VectorizedSampler(
                    env=post_eval["env"],
                    policy=post_eval["policy"],
                    n_envs=post_eval["n_envs"],
                )

        self.post_evals = post_evals

    def init_opt(self):
        obs_var = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1,
            flatten=False,
        )
        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1,
        )
        advantage_var = tf.placeholder(
            dtype=tf.float32,
            shape=(None,),
            name='advantage',
        )
        returns_var = tf.placeholder(
            dtype=tf.float32,
            shape=(None,),
            name='returns',
        )
        learning_rate = tf.placeholder(dtype=tf.float32, shape=tuple(), name='learning_rate')
        dist = self.policy.distribution
        old_dist_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] + list(shape), name='old_%s' % k)
            for k, shape in dist.dist_info_specs
            }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

        state_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] + list(shape), name=k)
            for k, shape in self.policy.state_info_specs
            }
        state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]

        if hasattr(self.policy, "ac"):
            dist_info_vars = self.policy.ac.joint_dist_info_vf(obs_var, state_info_vars)
            vf_prediction = dist_info_vars.pop("_vs")
        else:
            dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)
            vf_prediction = self.vf.predict_sym(obs_var)

        logli = dist.log_likelihood_sym(action_var, dist_info_vars)

        entropy = dist.entropy_sym(dist_info_vars)

        policy_loss = - tf.reduce_mean(logli * advantage_var + self.entropy_coeff * entropy)
        vf_loss = tf.reduce_mean(tf.square(vf_prediction - returns_var))

        total_loss = self.policy_loss_coeff * policy_loss + self.vf_loss_coeff * vf_loss

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        params = list(set(self.policy.get_params(trainable=True) + self.vf.get_params(trainable=True)))

        train_op = optimizer.minimize(total_loss, var_list=params)

        target_dist_info_vars = self.target_policy.dist_info_sym(obs_var, state_info_vars)
        mean_kl = tf.reduce_mean(dist.kl_sym(target_dist_info_vars, dist_info_vars))

        self.logging_infos = [
            ("Entropy", tf.reduce_mean(entropy)),
            ("Perplexity", tf.reduce_mean(tf.exp(entropy))),
            ("PolicyLoss", policy_loss),
            ("VfLoss", vf_loss),
            ("MeanKL", mean_kl),
        ]

        f_train = tensor_utils.compile_function(
            inputs=[obs_var, action_var, advantage_var, returns_var] + state_info_vars_list + old_dist_info_vars_list
                   + [learning_rate],
            outputs=[train_op] + [x[1] for x in self.logging_infos],
        )

        self.f_train = f_train

    def post_eval_policy(self, epoch):
        for post_eval in self.post_evals:
            label = post_eval["label"]
            sampler = post_eval["sampler"]
            batch_size = post_eval["batch_size"]
            max_path_length = post_eval["max_path_length"]
            with logger.prefix(label + " | "), logger.tabular_prefix(label + "|"):
                paths = sampler.obtain_samples(
                    epoch,
                    max_path_length=max_path_length,
                    batch_size=batch_size
                )
                returns = [sum(p["rewards"]) for p in paths]
                logger.record_tabular("NumTrajs", len(paths))
                logger.record_tabular_misc_stat('Return', returns, placement='front')

    def train(self):

        self.init_opt()

        time_logger = TimeLogger()

        with tf.Session() as sess:

            sess.run(tf.initialize_all_variables())

            for post_eval in self.post_evals:
                post_eval["sampler"].start_worker()

            if getattr(self.env, 'vectorized', False):
                vec_env = self.env.vec_env_executor(n_envs=self.n_envs)
            else:
                envs = [pickle.loads(pickle.dumps(self.env)) for _ in range(self.n_envs)]
                vec_env = VecEnvExecutor(envs=envs)

            dones = np.asarray([True] * self.n_envs)
            observations = vec_env.reset()

            past_rewards = []
            past_dones = []
            past_observations = []
            past_actions = []
            past_env_infos = []
            past_agent_infos = []
            past_vf_predictions = []
            running_returns = np.zeros((self.n_envs,))
            running_trajlens = np.zeros((self.n_envs,))
            all_trajlens = []
            all_returns = []

            logging_vals = dict()

            parallel_t = 0
            total_t = 0
            t_start = 0

            last_epoch_t = 0
            epoch_idx = 0

            progbar = pyprind.ProgBar(iterations=self.epoch_length)

            self.target_policy.set_param_values(self.policy.get_param_values())

            tf.get_default_graph().finalize()

            while True:
                parallel_t += 1
                total_t += self.n_envs

                self.policy.reset(dones)

                with time_logger.time("PolicyExecTime"):
                    actions, agent_infos = self.policy.get_actions(observations)
                    vf_predictions = agent_infos.pop("_vs")
                with time_logger.time("EnvStepTime"):
                    next_observations, rewards, dones, env_infos = vec_env.step(actions, max_path_length=np.inf)

                running_returns = running_returns + rewards
                running_trajlens = running_trajlens + 1

                if np.any(dones):
                    all_returns.extend(running_returns[dones])
                    all_trajlens.extend(running_trajlens[dones])

                    running_returns[dones] = 0
                    running_trajlens[dones] = 0

                if self.clip_rewards is not None:
                    past_rewards.append(np.clip(np.asarray(rewards), *self.clip_rewards))
                else:
                    past_rewards.append(np.asarray(rewards))
                past_dones.append(np.asarray(dones))
                past_observations.append(observations)
                past_actions.append(actions)
                past_env_infos.append(env_infos)
                past_agent_infos.append(agent_infos)
                past_vf_predictions.append(vf_predictions)

                if parallel_t - t_start >= self.t_max:
                    with time_logger.time("TrainPrepareTime"):
                        # first, compute all the returns
                        with time_logger.time("ComputeAdvantageTime"):
                            with time_logger.time("PredictTime"):
                                returns = self.vf.predict(next_observations)
                            past_returns = [None] * (parallel_t - t_start)
                            past_advantages = [None] * (parallel_t - t_start)
                            past_values = [None] * (parallel_t - t_start)
                            for t in reversed(range(t_start, parallel_t)):
                                returns = returns * self.discount * (1. - past_dones[t - t_start]) + past_rewards[
                                    t - t_start]
                                with time_logger.time("PredictTime"):
                                    values = past_vf_predictions[t - t_start]
                                advantages = returns - values
                                past_returns[t - t_start] = returns
                                past_advantages[t - t_start] = advantages
                                past_values[t - t_start] = values

                        with time_logger.time("ReshapeTime"):
                            vec_obs = np.concatenate(past_observations)
                            vec_actions = self.env.action_space.flatten_n(np.concatenate(past_actions))
                            vec_returns = tensor_utils.concat_tensor_list(past_returns)
                            vec_advantages = tensor_utils.concat_tensor_list(past_advantages)
                            vec_agent_infos = tensor_utils.concat_tensor_dict_list(past_agent_infos)

                            state_info_list = [vec_agent_infos[k] for k in self.policy.state_info_keys]
                            old_dist_info_list = [vec_agent_infos[k] for k in self.policy.distribution.dist_info_keys]

                            learning_rate = (self.max_steps - total_t - 1) * 1.0 / self.max_steps * self.learning_rate

                            all_inputs = [vec_obs, vec_actions, vec_advantages,
                                          vec_returns] + state_info_list + old_dist_info_list + [learning_rate]

                    with time_logger.time("TrainTime"):
                        _, *step_logging_vals = self.f_train(*all_inputs)
                        for (k, _), vals in zip(self.logging_infos, step_logging_vals):
                            if k not in logging_vals:
                                logging_vals[k] = []
                            logging_vals[k].append(vals)

                    past_rewards = []
                    past_dones = []
                    past_observations = []
                    past_actions = []
                    past_env_infos = []
                    past_agent_infos = []
                    past_vf_predictions = []
                    t_start = parallel_t

                observations = next_observations

                progbar.update(self.n_envs)

                if total_t - last_epoch_t >= self.epoch_length:
                    if progbar.active:
                        progbar.stop()

                    logger.record_tabular('Epoch', epoch_idx)

                    last_epoch_t = total_t
                    logger.record_tabular('NumTrajs', len(all_returns))
                    logger.record_tabular_misc_stat('Return', all_returns, placement='front')
                    logger.record_tabular_misc_stat('TrajLen', all_trajlens, placement='front')

                    for key, _ in self.logging_infos:
                        logger.record_tabular(key, np.mean(logging_vals[key]))

                    time_logger.dump_log()
                    time_logger.reset()

                    self.post_eval_policy(epoch_idx)

                    logger.dump_tabular(with_prefix=False)
                    all_returns = []
                    all_trajlens = []
                    logging_vals = dict()

                    epoch_idx += 1

                    if epoch_idx >= self.n_epochs:
                        break
                    else:
                        progbar = pyprind.ProgBar(iterations=self.epoch_length)

                    logger.log("Saving snapshot...")
                    snapshot_params = dict(
                        algo=self,
                        policy=self.policy,
                        vf=self.vf,
                    )
                    logger.save_itr_params(epoch_idx, snapshot_params)

                    self.target_policy.set_param_values(self.policy.get_param_values())


if __name__ == "__main__":
    from sandbox.rocky.neural_learner.envs.parallel_atari_env import AtariEnv

    logger.log("creating env")
    env = TfEnv(AtariEnv(game="breakout", obs_type="ram"))
    logger.log("creating test env")
    logger.log("actor critic")
    actor_critic = JointActorCritic(env_spec=env.spec)
    policy = actor_critic.policy
    vf = actor_critic.vf

    algo = SAAC(env=env, policy=policy, vf=vf, n_envs=16, t_max=5, epoch_length=1000, n_epochs=2500)
    logger.log("training")
    algo.train()
