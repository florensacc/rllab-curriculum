from rllab.algos.base import RLAlgorithm
from rllab.misc.overrides import overrides
from rllab.misc import special
from rllab.misc import ext
from rllab.sampler import parallel_sampler
from rllab.plotter import plotter
from functools import partial
import rllab.misc.logger as logger
import theano.tensor as TT
import cPickle as pickle
import numpy as np
import pyprind
import theano
import lasagne


def parse_update_method(update_method, **kwargs):
    if update_method == 'adam':
        return partial(lasagne.updates.adam, **ext.compact(kwargs))
    elif update_method == 'sgd':
        return partial(lasagne.updates.sgd, **ext.compact(kwargs))
    else:
        raise NotImplementedError


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


class DPG(RLAlgorithm):
    """
    Deterministic Policy Gradient.
    """

    def __init__(
            self,
            batch_size=32,
            n_epochs=200,
            epoch_length=1000,
            min_pool_size=10000,
            replay_pool_size=1000000,
            discount=0.99,
            max_path_length=250,
            qf_weight_decay=0.,
            qf_update_method='adam',
            qf_learning_rate=1e-3,
            policy_weight_decay=0,
            policy_update_method='adam',
            policy_learning_rate=1e-3,
            eval_samples=10000,
            eval_whole_paths=True,
            soft_target=True,
            soft_target_tau=0.001,
            hard_target_interval=1000,
            n_updates_per_sample=1,
            normalize_yval=False,
            normalize_qval=False,
            running_y_alpha=0.1,
            running_q_alpha=0.1,
            scale_reward=1.0,
            include_horizon_terminal_transitions=False,
            plot=False):
        """

        :param batch_size: Number of samples for each minibatch.
        :param n_epochs: Number of epochs. Policy will be evaluated after each epoch.
        :param epoch_length: How many timesteps for each epoch.
        :param min_pool_size: Minimum size of the pool to start training.
        :param replay_pool_size: Size of the experience replay pool.
        :param discount: Discount factor for the cumulative return.
        :param max_path_length: Discount factor for the cumulative return.
        :param qf_weight_decay: Weight decay factor for parameters of the Q function.
        :param qf_update_method: Online optimization method for training Q function.
        :param qf_learning_rate: Learning rate for training Q function.
        :param policy_weight_decay: Weight decay factor for parameters of the policy.
        :param policy_update_method: Online optimization method for training the policy.
        :param policy_learning_rate: Learning rate for training the policy.
        :param eval_samples: Number of samples (timesteps) for evaluating the policy.
        :param eval_whole_paths: Whether to make sure that all trajectories are executed until the terminal state or
        the max_path_length, even at the expense of possibly more samples for evaluation.
        :param soft_target: Whether to use soft target updates.
        :param soft_target_tau: Interpolation parameter for doing the soft target update.
        :param n_updates_per_sample: Number of Q function and policy updates per new sample obtained
        :param scale_reward: The scaling factor applied to the rewards when training
        :param include_horizon_terminal_transitions: whether to include transitions with terminal=True because the
        horizon was reached. This might make the Q value back up less stable for certain tasks.
        :param plot: Whether to visualize the policy performance after each eval_interval.
        :return:
        """
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.epoch_length = epoch_length
        self.min_pool_size = min_pool_size
        self.replay_pool_size = replay_pool_size
        self.discount = discount
        self.max_path_length = max_path_length
        self.qf_weight_decay = qf_weight_decay
        self.qf_update_method = \
            parse_update_method(
                qf_update_method,
                learning_rate=qf_learning_rate,
            )
        self.qf_learning_rate = qf_learning_rate
        self.policy_weight_decay = policy_weight_decay
        self.policy_update_method = \
            parse_update_method(
                policy_update_method,
                learning_rate=policy_learning_rate,
            )
        self.policy_learning_rate = policy_learning_rate
        self.eval_samples = eval_samples
        self.eval_whole_paths = eval_whole_paths
        self.soft_target = soft_target
        self.soft_target_tau = soft_target_tau
        self.hard_target_interval = hard_target_interval
        self.n_updates_per_sample = n_updates_per_sample
        self._include_horizon_terminal_transitions = include_horizon_terminal_transitions
        self.plot = plot

        self.qf_loss_averages = []
        self.policy_surr_averages = []
        # self.obses = []
        # self.actions = []
        self.q_averages = []
        self.y_averages = []
        self.paths = []
        self.es_path_returns = []
        self.paths_samples_cnt = 0

        self._scale_reward = scale_reward

        self._normalize_yval = normalize_yval
        self._normalize_qval = normalize_qval
        self._q_mean = theano.shared(0., "q_mean")
        self._q_std = theano.shared(10000., "q_std")
        self._y_mean = theano.shared(0., "y_mean")
        self._y_std = theano.shared(10000., "y_std")
        # self._q_stats_computed = False
        self._running_q_alpha = running_q_alpha
        self._running_y_alpha = running_y_alpha

    def start_worker(self, env, policy):
        parallel_sampler.populate_task(env, policy)
        if self.plot:
            plotter.init_plot(env, policy)

    @overrides
    def train(self, env, policy, qf, es, **kwargs):
        # This seems like a rather sequential method
        pool = SimpleReplayPool(
            max_pool_size=self.replay_pool_size,
            observation_dim=env.observation_space.flat_dim,
            action_dim=env.action_space.flat_dim,
        )
        self.start_worker(env, policy)

        opt_info = self.init_opt(env, policy, qf)
        itr = 0
        path_length = 0
        path_return = 0
        terminal = False
        observation = env.reset()

        sample_policy = pickle.loads(pickle.dumps(policy))

        for epoch in xrange(self.n_epochs):
            logger.push_prefix('epoch #%d | ' % epoch)
            logger.log("Training started")
            for epoch_itr in pyprind.prog_bar(xrange(self.epoch_length)):
                # Execute policy
                if terminal:  # or path_length > self.max_path_length:
                    # Note that if the last time step ends an episode, the very
                    # last state and observation will be ignored and not added
                    # to the replay pool
                    observation = env.reset()
                    es.reset()
                    sample_policy.reset()
                    self.es_path_returns.append(path_return)
                    path_length = 0
                    path_return = 0
                action = es.get_action(itr, observation, policy=sample_policy)  # qf=qf)

                next_observation, reward, terminal, _ = env.step(action)
                path_length += 1
                path_return += reward

                if not terminal and path_length >= self.max_path_length:
                    terminal = True
                    # only include the terminal transition in this case if the flag was set
                    if self._include_horizon_terminal_transitions:
                        pool.add_sample(observation, action, reward * self._scale_reward, terminal)
                else:
                    pool.add_sample(observation, action, reward * self._scale_reward, terminal)

                observation = next_observation

                if pool.size >= self.min_pool_size:
                    for update_itr in xrange(self.n_updates_per_sample):
                        # Train policy
                        batch = pool.random_batch(self.batch_size)
                        opt_info = self.do_training(
                            itr, batch, qf, policy, opt_info)
                    sample_policy.set_param_values(policy.get_param_values())

                itr += 1

            logger.log("Training finished")
            if pool.size >= self.min_pool_size:
                opt_info = self.evaluate(epoch, qf, policy, opt_info, env, pool)
                yield opt_info
                params = self.get_epoch_snapshot(
                    epoch, env, qf, policy, es, opt_info)
                logger.save_itr_params(epoch, params)
            logger.dump_tabular(with_prefix=False)
            logger.pop_prefix()

    def init_opt(self, env, policy, qf):

        # First, create "target" policy and Q functions
        target_policy = pickle.loads(pickle.dumps(policy))
        target_qf = pickle.loads(pickle.dumps(qf))

        # y need to be computed first
        obs = env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1,
        )
        # next_obs = new_tensor(
        #    'next_obs',
        #    ndim=1+len(env.observation_shape),
        #    dtype=env.observation_dtype
        # )

        # rewards = TT.vector('rewards')
        # terminals = TT.vector('terminals')

        # The yi values are computed separately as above and then passed to
        # the training functions below
        action = env.action_space.new_tensor_variable(
            'action',
            extra_dims=1,
        )
        yvar = TT.vector('ys')

        qf_weight_decay_term = 0.5 * self.qf_weight_decay * \
                               sum([TT.sum(TT.square(param)) for param in
                                    qf.get_params(regularizable=True)])

        qval = qf.get_qval_sym(obs, action)

        qf_loss = TT.mean(TT.square(yvar - qval))
        if self._normalize_yval:
            qf_loss = qf_loss / (self._y_std ** 2)
        qf_reg_loss = qf_loss + qf_weight_decay_term

        policy_weight_decay_term = 0.5 * self.policy_weight_decay * \
                                   sum([TT.sum(TT.square(param))
                                        for param in policy.get_params(regularizable=True)])
        policy_qval = qf.get_qval_sym(
            obs, policy.get_action_sym(obs),
            deterministic=True
        )
        if self._normalize_qval:
            policy_qval = (policy_qval - self._q_mean) / self._q_std
        policy_surr = -TT.mean(policy_qval)

        policy_reg_surr = policy_surr + policy_weight_decay_term

        qf_updates = self.qf_update_method(
            qf_reg_loss, qf.get_params(trainable=True))
        policy_updates = self.policy_update_method(
            policy_reg_surr, policy.get_params(trainable=True))

        f_train_qf = ext.compile_function(
            inputs=[yvar, obs, action],
            outputs=[qf_loss, qval],
            updates=qf_updates
        )

        f_train_policy = ext.compile_function(
            inputs=[obs],
            outputs=policy_surr,
            updates=policy_updates
        )

        # target_updates = []
        # for p, tp in zip(
        #        qf.get_params() + policy.get_params(),
        #        target_qf.get_params() + target_policy.get_params()):
        #    if self.soft_target:
        #        target_updates.append((tp, self.soft_target_tau * p +
        #                               (1 - self.soft_target_tau) * tp))
        #    else:
        #        target_updates.append((tp, p))

        # f_update_targets = compile_function(
        #    inputs=[],
        #    outputs=[],
        #    updates=target_updates,
        # )

        return dict(
            # f_y=f_y,
            f_train_qf=f_train_qf,
            f_train_policy=f_train_policy,
            # f_update_targets=f_update_targets,
            target_qf=target_qf,
            target_policy=target_policy,
        )

    def do_training(self, itr, batch, qf, policy, opt_info):

        obs, actions, rewards, next_obs, terminals = ext.extract(
            batch,
            "observations", "actions", "rewards", "next_observations",
            "terminals"
        )

        # compute the on-policy y values
        target_qf = opt_info["target_qf"]
        target_policy = opt_info["target_policy"]

        next_actions, _ = target_policy.get_actions(next_obs)
        next_qvals = target_qf.get_qval(next_obs, next_actions)

        ys = rewards + (1 - terminals) * self.discount * next_qvals

        batch_y_mean = np.mean(ys)
        batch_y_std = np.std(ys)
        self._y_mean.set_value(self._y_mean.get_value() * (1 - self._running_y_alpha) + batch_y_mean *
                               self._running_y_alpha)
        self._y_std.set_value(self._y_std.get_value() * (1 - self._running_y_alpha) + batch_y_std *
                              self._running_y_alpha)

        if self._normalize_yval:
            ys = (ys - self._y_mean.get_value()) / self._y_std.get_value()

        # print sum(terminals)

        # if itr % 100 == 0:
        #    import ipdb; ipdb.set_trace()

        f_train_qf = opt_info["f_train_qf"]
        f_train_policy = opt_info["f_train_policy"]
        # f_update_targets = opt_info["f_update_targets"]

        qf_loss, qval = f_train_qf(ys, obs, actions)

        batch_q_mean = np.mean(qval)
        batch_q_std = np.std(qval)
        self._q_mean.set_value(self._q_mean.get_value() * (1 - self._running_q_alpha) + batch_q_mean *
                               self._running_q_alpha)
        self._q_std.set_value(self._q_std.get_value() * (1 - self._running_q_alpha) + batch_q_std *
                              self._running_q_alpha)

        policy_surr = f_train_policy(obs)

        if self.soft_target:
            target_policy.set_param_values(
                target_policy.get_param_values() * (1.0 - self.soft_target_tau) +
                policy.get_param_values() * self.soft_target_tau)
            target_qf.set_param_values(
                target_qf.get_param_values() * (1.0 - self.soft_target_tau) +
                qf.get_param_values() * self.soft_target_tau)

        elif itr % self.hard_target_interval == 0:
            raise NotImplementedError
            # f_update_targets()

        self.qf_loss_averages.append(qf_loss)
        self.policy_surr_averages.append(policy_surr)
        # self.obses.append(obs)
        # self.actions.append(actions)
        self.q_averages.append(qval)
        self.y_averages.append(ys)

        # print len(self.q_averages), len(self.y_averages)
        # if len(self.q_averages) != len(self.y_averages):
        #     import ipdb; ipdb.set_trace()


        return opt_info

    def evaluate(self, epoch, qf, policy, opt_info, env, pool):

        logger.log("Collecting samples for evaluation")

        try:
            parallel_sampler.request_samples(
                policy_params=policy.get_param_values(),
                max_samples=self.eval_samples,
                max_path_length=self.max_path_length,
                whole_paths=self.eval_whole_paths,
            )
        except Exception as e:
            print e
            import ipdb; ipdb.set_trace()

        paths = parallel_sampler.collect_paths()

        # obs = np.vstack([p["observations"] for p in paths])
        # actions = np.vstack([p["actions"] for p in paths])

        # qvals = qf.get_qval(obs, actions)
        # self._q_mean.set_value(np.mean(qvals))
        # self._q_std.set_value(np.std(qvals))
        # self._q_stats_computed = True

        average_discounted_return = np.mean(
            [special.discount_return(path["rewards"], self.discount) for path in paths]
        )

        returns = [sum(path["rewards"]) for path in paths]

        # if len(self.q_averages) > 0:
        all_qs = np.concatenate(self.q_averages)
        all_ys = np.concatenate(self.y_averages)
        # else:
        #     all_qs = np.array([])
        #     all_ys = np.array([])

        average_q_loss = np.mean(self.qf_loss_averages)
        average_policy_surr = np.mean(self.policy_surr_averages)
        average_action = np.mean(np.square(np.concatenate(
            [path["actions"] for path in paths]
        )))

        policy_reg_param_norm = np.linalg.norm(
            policy.get_param_values(regularizable=True)
        )
        qfun_reg_param_norm = np.linalg.norm(
            qf.get_param_values(regularizable=True)
        )

        logger.record_tabular('Epoch', epoch)
        logger.record_tabular('AverageReturn',
                              np.mean(returns))
        logger.record_tabular('StdReturn',
                              np.std(returns))
        logger.record_tabular('MaxReturn',
                              np.max(returns))
        logger.record_tabular('MinReturn',
                              np.min(returns))
        if len(self.es_path_returns) > 0:
            logger.record_tabular('AverageEsReturn',
                                  np.mean(self.es_path_returns))
            logger.record_tabular('StdEsReturn',
                                  np.std(self.es_path_returns))
            logger.record_tabular('MaxEsReturn',
                                  np.max(self.es_path_returns))
            logger.record_tabular('MinEsReturn',
                                  np.min(self.es_path_returns))
        logger.record_tabular('AverageDiscountedReturn',
                              average_discounted_return)
        logger.record_tabular('AverageQLoss', average_q_loss)
        logger.record_tabular('AveragePolicySurr', average_policy_surr)
        logger.record_tabular('AverageQ', np.mean(all_qs))
        logger.record_tabular('AverageAbsQ', np.mean(np.abs(all_qs)))
        logger.record_tabular('AverageY', np.mean(all_ys))
        logger.record_tabular('AverageAbsY', np.mean(np.abs(all_ys)))
        logger.record_tabular('AverageAbsQYDiff',
                              np.mean(np.abs(all_qs - all_ys)))
        logger.record_tabular('AverageAction', average_action)

        logger.record_tabular('PolicyRegParamNorm',
                              policy_reg_param_norm)
        logger.record_tabular('QFunRegParamNorm',
                              qfun_reg_param_norm)
        logger.record_tabular('QMean', self._q_mean.get_value())
        logger.record_tabular('QStd', self._q_std.get_value())
        logger.record_tabular('YMean', self._y_mean.get_value())
        logger.record_tabular('YStd', self._y_std.get_value())

        env.log_diagnostics(paths)

        self.qf_loss_averages = []
        self.policy_surr_averages = []

        self.q_averages = []
        self.y_averages = []
        self.es_path_returns = []

        return opt_info#ext.merge_dict(opt_info, dict(
        #    eval_paths=paths,
        #))

    def get_epoch_snapshot(self, epoch, env, qf, policy, es, opt_info):
        return dict(
            env=env,
            epoch=epoch,
            qf=qf,
            policy=policy,
            target_qf=opt_info["target_qf"],
            target_policy=opt_info["target_policy"],
            es=es,
        )
