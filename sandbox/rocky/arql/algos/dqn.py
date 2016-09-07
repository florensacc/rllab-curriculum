


import numpy as np
import tensorflow as tf
from rllab.misc import logger
from rllab.misc.ext import AttrDict
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler
import pyprind


class SimpleReplayPool(object):
    def __init__(
            self, max_pool_size, observation_dim, action_dim):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_pool_size = max_pool_size
        self._observations = np.zeros(
            (max_pool_size, observation_dim),
        )
        self._actions = np.zeros(
            (max_pool_size, action_dim),
        )
        self._rewards = np.zeros(max_pool_size)
        self._terminals = np.zeros(max_pool_size, dtype='uint8')
        self._bottom = 0
        self._top = 0
        self._size = 0

    def add_sample(self, observation, action, reward, terminal):
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._top = (self._top + 1) % self._max_pool_size
        if self._size >= self._max_pool_size:
            self._bottom = (self._bottom + 1) % self._max_pool_size
        else:
            self._size += 1

    def random_batch(self, batch_size):
        assert self._size > batch_size
        indices = np.zeros(batch_size, dtype='uint64')
        transition_indices = np.zeros(batch_size, dtype='uint64')
        count = 0
        while count < batch_size:
            index = np.random.randint(self._bottom, self._bottom + self._size) % self._max_pool_size
            # make sure that the transition is valid: if we are at the end of the pool, we need to discard
            # this sample
            if index == self._size - 1 and self._size <= self._max_pool_size:
                continue
            # if self._terminals[index]:
            #     continue
            transition_index = (index + 1) % self._max_pool_size
            indices[count] = index
            transition_indices[count] = transition_index
            count += 1
        return dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._observations[transition_indices]
        )

    @property
    def size(self):
        return self._size


class DQN(object):
    def __init__(
            self,
            env,
            qf,
            es,
            target_qf,
            tf_optimizer=None,
            double_dqn=True,
            replay_pool_size=1000000,
            min_pool_size=10000,
            update_interval=1,
            target_update_interval=10000,
            max_path_length=500,
            eval_max_path_length=500,
            learning_rate=1e-4,
            n_epochs=1000,
            epoch_length=1000,
            eval_samples=10000,
            batch_size=32,
            discount=0.99,
            scale_reward=1.,
    ):
        """
        :param env: Environment
        :param qf: Q function
        :param es: Exploration strategy
        :param tf_optimizer: Tensorflow optimizer to be used. By default it uses Adam
        :param double_dqn: Whether to use the double Q-learning formula
        :param replay_pool_size: Size of the replay pool
        :param min_pool_size: Minimum size of the replay pool to start learning
        :param learning_rate: Learning rate
        :param n_epochs: Number of epochs to perform training
        :param epoch_length: Number of time steps in each epoch
        :param batch_size: Size of each training minibatch
        :param discount: Discount factor
        """
        obs_dim = env.observation_space.flat_dim
        action_dim = env.action_space.flat_dim

        self.env = env
        self.qf = qf
        self.es = es
        self.target_qf = target_qf

        self.replay_pool = SimpleReplayPool(
            observation_dim=obs_dim,
            action_dim=action_dim,
            max_pool_size=replay_pool_size,
        )
        self.min_pool_size = min_pool_size
        self.max_path_length = max_path_length

        if tf_optimizer is None:
            tf_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.tf_optimizer = tf_optimizer
        self.double_dqn = double_dqn
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.update_interval = update_interval
        self.epoch_length = epoch_length
        self.batch_size = batch_size
        self.eval_samples = eval_samples
        self.discount = discount
        self.scale_reward = scale_reward
        self.target_update_interval = target_update_interval
        self.sampler = VectorizedSampler(AttrDict(
            batch_size=eval_samples,
            max_path_length=eval_max_path_length,
            env=self.env,
            policy=self.qf,
        ))
        self.f_train = None
        self.es_path_returns = []
        self.logging_info = []
        self.logged_columns = []

    def init_opt(self):

        obs_var = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1,
        )
        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1,
        )
        next_obs_var = self.env.observation_space.new_tensor_variable(
            'next_obs',
            extra_dims=1,
        )
        reward_var = tf.placeholder(dtype=tf.float32, shape=(None,), name="reward")
        terminal_var = tf.placeholder(dtype=tf.float32, shape=(None,), name="terminal")

        # compute max_a' Q(s',a')
        if self.double_dqn:
            q_next = self.target_qf.qval_sym(next_obs_var, self.qf.argmax_qval_sym(next_obs_var))
        else:
            q_next = self.target_qf.max_qval_sym(next_obs_var)

        target_var = reward_var + self.discount * (1 - terminal_var) * q_next
        q_var = self.qf.qval_sym(obs_var, action_var)

        loss = tf.reduce_mean(tf.square(q_var - tf.stop_gradient(target_var)))

        train_op = self.tf_optimizer.minimize(loss, var_list=self.qf.get_params(trainable=True))

        self.logging_info.extend([
            ("AverageLoss", loss),
            ("AverageAbsTargetQ", tf.reduce_mean(tf.abs(target_var))),
            ("AverageAbsQ", tf.reduce_mean(tf.abs(q_var))),
        ])

        f_train = tensor_utils.compile_function(
            inputs=[obs_var, action_var, next_obs_var, reward_var, terminal_var],
            outputs=[train_op] + [x[1] for x in self.logging_info],
        )

        self.f_train = f_train

    def train(self):
        self.init_opt()

        observation_space = self.env.observation_space
        action_space = self.env.action_space

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            self.target_qf.set_param_values(self.qf.get_param_values())
            self.sampler.start_worker()

            obs = self.env.reset()
            self.es.reset()
            self.qf.reset()
            global_itr = 0
            path_length = 0
            path_return = 0
            done = False

            for epoch in range(self.n_epochs):

                logger.log("Starting epoch %d" % epoch)
                logger.record_tabular('Epoch', epoch)

                self.es_path_returns = []
                self.logged_columns = []

                for epoch_itr in pyprind.prog_bar(range(self.epoch_length)):
                    if done:
                        obs = self.env.reset()
                        self.es.reset()
                        self.qf.reset()
                        self.es_path_returns.append(path_return)
                        path_length = 0
                        path_return = 0

                    action = self.es.get_action(iteration=global_itr, observation=obs, policy=self.qf)
                    next_obs, reward, done, info = self.env.step(action)
                    path_length += 1
                    path_return += reward

                    flat_obs = observation_space.flatten(obs)
                    flat_action = action_space.flatten(action)

                    self.replay_pool.add_sample(
                        observation=flat_obs,
                        action=flat_action,
                        reward=reward * self.scale_reward,
                        terminal=done
                    )

                    if not done and path_length >= self.max_path_length:
                        done = True

                    obs = next_obs

                    if self.replay_pool.size >= self.min_pool_size:
                        batch = self.replay_pool.random_batch(self.batch_size)
                        self.do_training(batch)

                    global_itr += 1

                    if global_itr % self.target_update_interval == 0:
                        self.target_qf.set_param_values(self.qf.get_param_values())

                if len(self.logged_columns) > 0:
                    for (key, _), val in zip(self.logging_info, np.asarray(self.logged_columns).mean(axis=0)):
                        logger.record_tabular(key, val)
                else:
                    for key, _ in self.logging_info:
                        logger.record_tabular(key, np.nan)

                paths = self.sampler.obtain_samples(epoch)
                returns = np.array([np.sum(p["rewards"]) for p in paths])

                logger.record_tabular('NSamples', global_itr)
                logger.record_tabular('AverageReturn', np.mean(returns))
                logger.record_tabular('StdReturn', np.std(returns))
                logger.record_tabular('MaxReturn', np.max(returns))
                logger.record_tabular('MinReturn', np.min(returns))

                logger.record_tabular('AverageEsReturn', np.mean(self.es_path_returns))

                logger.dump_tabular()

            self.sampler.shutdown_worker()

    def do_training(self, batch):
        batch_obs = batch["observations"]
        batch_actions = batch["actions"]
        batch_next_obs = batch["next_observations"]
        batch_rewards = batch["rewards"]
        batch_terminals = batch["terminals"]
        train_results = self.f_train(batch_obs, batch_actions, batch_next_obs, batch_rewards, batch_terminals)[1:]
        self.logged_columns.append(train_results)
