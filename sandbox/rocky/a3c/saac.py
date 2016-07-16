from __future__ import print_function
from __future__ import absolute_import

# Synchronous Advantage Actor-Critic
import tensorflow as tf
import sandbox.rocky.tf.core.layers as L
from sandbox.rocky.tf.core.layers_powered import LayersPowered
import numpy as np

from sandbox.rocky.tf.policies.base import StochasticPolicy
from sandbox.rocky.tf.misc import tensor_utils
from rllab.core.serializable import Serializable
from sandbox.rocky.tf.distributions.categorical import Categorical
from sandbox.rocky.tf.envs.parallel_vec_env_executor import ParallelVecEnvExecutor
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.misc import logger
from rllab.misc import special
from contextlib import contextmanager
import time


class PolicyWrapper(StochasticPolicy, LayersPowered, Serializable):
    def __init__(self, env_spec, ac):
        super(PolicyWrapper, self).__init__(env_spec)
        Serializable.quick_init(self, locals())
        self.ac = ac
        self.f_dist = tensor_utils.compile_function(
            inputs=[self.ac.l_in.input_var],
            outputs=self.dist_info_sym(self.ac.l_in.input_var, dict())
        )
        LayersPowered.__init__(self, [self.ac.l_prob])

    @property
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
        return actions[0], {k: v[0] for k, v in agent_infos.iteritems()}

    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        dist_info = self.f_dist(flat_obs)
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
        flat_obs = self.observation_space.flatten_n(observations)
        return self.f_predict(flat_obs)

    def predict_sym(self, obs_var):
        return tf.reshape(L.get_output(self.ac.l_v, {self.ac.l_in: obs_var}), (-1,))


class JointActorCritic(LayersPowered):
    def __init__(self, env_spec, hidden_nonlinearity=tf.nn.relu, W=L.he_init, b=tf.constant_initializer(0.1)):
        with tf.variable_scope("dqn_head"):
            obs_dim = env_spec.observation_space.flat_dim
            action_dim = env_spec.action_space.flat_dim
            l_in = L.InputLayer(
                shape=(None, obs_dim),
                name="input"
            )
            l_nonflat_in = L.reshape(l_in, shape=([0],) + env_spec.observation_space.shape, name="input_nonflat")
            l_in_reshaped = L.dimshuffle(l_nonflat_in, (0, 2, 3, 1), name="input_reshaped")
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
        for k, vs in self.stats.iteritems():
            logger.record_tabular(k, np.sum(vs))

    def reset(self):
        self.stats = {}


class SAAC(object):
    def __init__(self,
                 env,
                 policy,
                 vf,
                 n_parallel=8,
                 test_env=None,
                 t_max=5,
                 discount=0.99,
                 policy_loss_coeff=1.,
                 vf_loss_coeff=0.5,
                 entropy_coeff=0.01,
                 n_epochs=1000,
                 epoch_length=1000,
                 # max_steps=8 * 10 ** 7,
                 learning_rate=7e-4,
                 ):
        """
        :type env: Env
        :type test_env: Env
        :type policy: StochasticPolicy
        """
        self.env = env
        self.policy = policy
        if test_env is None:
            test_env = env
        self.test_env = test_env
        self.vf = vf
        self.n_parallel = n_parallel
        self.t_max = t_max
        self.discount = discount
        self.policy_loss_coeff = policy_loss_coeff
        self.vf_loss_coeff = vf_loss_coeff
        self.entropy_coeff = entropy_coeff
        self.n_epochs = n_epochs
        self.epoch_length = epoch_length
        self.max_steps = n_epochs * epoch_length
        self.learning_rate = learning_rate

    def init_opt(self):
        obs_var = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1,
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

        dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)

        logli = dist.log_likelihood_sym(action_var, dist_info_vars)

        entropy = dist.entropy_sym(dist_info_vars)

        policy_loss = - tf.reduce_mean(logli * advantage_var + self.entropy_coeff * entropy)
        vf_loss = tf.reduce_mean(tf.square(self.vf.predict_sym(obs_var) - returns_var))

        total_loss = self.policy_loss_coeff * policy_loss + self.vf_loss_coeff * vf_loss

        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, epsilon=1e-1, decay=0.99)

        params = list(set(self.policy.get_params(trainable=True) + self.vf.get_params(trainable=True)))

        train_op = optimizer.minimize(total_loss, var_list=params)

        f_train = tensor_utils.compile_function(
            inputs=[obs_var, action_var, advantage_var, returns_var] + state_info_vars_list + old_dist_info_vars_list
                   + [learning_rate],
            outputs=[train_op],
        )

        self.f_train = f_train

    def train(self):
        self.init_opt()

        time_logger = TimeLogger()
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            vec_env = ParallelVecEnvExecutor(self.env, n=self.n_parallel, max_path_length=np.inf)

            dones = [True] * self.n_parallel
            observations = vec_env.reset()

            past_rewards = []
            past_dones = []
            past_observations = []
            past_actions = []
            past_agent_infos = []
            past_env_infos = []
            running_returns = np.zeros((self.n_parallel,))
            all_returns = []

            parallel_t = 0
            total_t = 0
            t_start = 0

            last_epoch_t = 0

            while True:
                parallel_t += 1
                total_t += self.n_parallel

                self.policy.reset(dones)

                with time_logger.time("PolicyExecTime"):
                    actions, agent_infos = self.policy.get_actions(observations)
                with time_logger.time("EnvStepTime"):
                    next_observations, rewards, dones, env_infos = vec_env.step(actions)

                running_returns = running_returns + rewards

                for idx, done in enumerate(dones):
                    if done:
                        all_returns.append(running_returns[idx])
                        running_returns[idx] = 0

                past_rewards.append(np.asarray(rewards))
                past_dones.append(np.asarray(dones))
                past_observations.append(observations)
                past_actions.append(actions)
                past_env_infos.append(env_infos)
                past_agent_infos.append(agent_infos)

                if parallel_t - t_start >= self.t_max:
                    with time_logger.time("TrainPrepareTime"):
                        # first, compute all the returns
                        with time_logger.time("ComputeAdvantageTime"):
                            with time_logger.time("PredictTime"):
                                returns = self.vf.predict(next_observations)
                            past_returns = [None] * (parallel_t - t_start)
                            past_advantages = [None] * (parallel_t - t_start)
                            past_values = [None] * (parallel_t - t_start)
                            for t in reversed(xrange(t_start, parallel_t)):
                                returns = returns * self.discount * (1. - past_dones[t - t_start]) + past_rewards[
                                    t - t_start]
                                with time_logger.time("PredictTime"):
                                    values = self.vf.predict(past_observations[t - t_start])
                                advantages = returns - values
                                past_returns[t - t_start] = returns
                                past_advantages[t - t_start] = advantages
                                past_values[t - t_start] = values

                        with time_logger.time("ReshapeTime"):
                            vec_obs = self.env.observation_space.flatten_n(sum(past_observations, []))
                            vec_actions = self.env.action_space.flatten_n(sum(map(list, past_actions), []))
                            vec_returns = tensor_utils.concat_tensor_list(past_returns)
                            vec_advantages = tensor_utils.concat_tensor_list(past_advantages)
                            vec_agent_infos = tensor_utils.concat_tensor_dict_list(past_agent_infos)

                            state_info_list = [vec_agent_infos[k] for k in self.policy.state_info_keys]
                            old_dist_info_list = [vec_agent_infos[k] for k in self.policy.distribution.dist_info_keys]

                            learning_rate = (self.max_steps - total_t - 1) * 1.0 / self.max_steps * self.learning_rate

                            all_inputs = [vec_obs, vec_actions, vec_advantages,
                                          vec_returns] + state_info_list + old_dist_info_list + [learning_rate]

                            logger.log("training... t: %d" % total_t)

                    with time_logger.time("TrainTime"):
                        self.f_train(*all_inputs)

                    past_rewards = []
                    past_dones = []
                    past_observations = []
                    past_actions = []
                    past_env_infos = []
                    past_agent_infos = []
                    t_start = parallel_t

                observations = next_observations

                if total_t - last_epoch_t >= self.epoch_length:
                    last_epoch_t = total_t
                    if len(all_returns) == 0:
                        logger.record_tabular('AverageReturn', np.nan)
                        logger.record_tabular('MaxReturn', np.nan)
                        logger.record_tabular('MinReturn', np.nan)
                        logger.record_tabular('StdReturn', np.nan)
                    else:
                        logger.record_tabular('AverageReturn', np.mean(all_returns))
                        logger.record_tabular('MaxReturn', np.max(all_returns))
                        logger.record_tabular('MinReturn', np.min(all_returns))
                        logger.record_tabular('StdReturn', np.std(all_returns))
                    logger.record_tabular('NumTrajs', len(all_returns))
                    time_logger.dump_log()
                    time_logger.reset()
                    logger.dump_tabular()
                    all_returns = []

                    # vec_env
                    # if done:
                    #     episode_r = 0
            obs = self.env.reset()
            action, agent_info = self.policy.get_action(obs)
            import ipdb;
            ipdb.set_trace()
            pass


if __name__ == "__main__":
    from sandbox.rocky.a3c.ale import ImageAtariEnv
    from rllab.sampler import parallel_sampler

    parallel_sampler.initialize(8)

    logger.log("creating env")
    env = TfEnv(ImageAtariEnv("pong"))
    logger.log("creating test env")
    test_env = TfEnv(ImageAtariEnv("pong", treat_life_lost_as_terminal=False))
    logger.log("actor critic")
    actor_critic = JointActorCritic(env_spec=env.spec)
    policy = actor_critic.policy  # Nips
    vf = actor_critic.vf

    algo = SAAC(env=env, test_env=test_env, policy=policy, vf=vf, n_parallel=16)
    logger.log("training")
    algo.train()
