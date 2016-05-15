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

# exploration imports
# -------------------
import theano
import lasagne
from collections import deque
import time
from sandbox.rein.dynamics_models.nn_uncertainty import vbnn
# ------------------


def parse_update_method(update_method, **kwargs):
    if update_method == 'adam':
        return partial(lasagne.updates.adam, **ext.compact(kwargs))
    elif update_method == 'sgd':
        return partial(lasagne.updates.sgd, **ext.compact(kwargs))
    else:
        raise NotImplementedError


class DynamicsSimpleReplayPool(object):
    """Replay pool"""

    def __init__(
            self, max_pool_size, observation_shape, action_dim,
            observation_dtype=theano.config.floatX,  # @UndefinedVariable
            action_dtype=theano.config.floatX):  # @UndefinedVariable
        self._observation_shape = observation_shape
        self._action_dim = action_dim
        self._observation_dtype = observation_dtype
        self._action_dtype = action_dtype
        self._max_pool_size = max_pool_size

        self._observations = np.zeros(
            (max_pool_size,) + observation_shape,
            dtype=observation_dtype
        )
        self._actions = np.zeros(
            (max_pool_size, action_dim),
            dtype=action_dtype
        )
        self._rewards = np.zeros(max_pool_size, dtype='float32')
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
            self._size = self._size + 1

    def random_batch(self, batch_size):
        assert self._size > batch_size
        indices = np.zeros(batch_size, dtype='uint64')
        transition_indices = np.zeros(batch_size, dtype='uint64')
        count = 0
        while count < batch_size:
            index = np.random.randint(
                self._bottom, self._bottom + self._size) % self._max_pool_size
            # make sure that the transition is valid: if we are at the end of the pool, we need to discard
            # this sample
            if index == self._size - 1 and self._size <= self._max_pool_size:
                continue
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

    def mean_obs_act(self):
        if self._size >= self._max_pool_size:
            obs = self._observations
            act = self._actions
        else:
            obs = self._observations[:self._top + 1]
            act = self._actions[:self._top + 1]
        obs_mean = np.mean(obs, axis=0)
        obs_std = np.std(obs, axis=0)
        act_mean = np.mean(act, axis=0)
        act_std = np.std(act, axis=0)
        return obs_mean, obs_std, act_mean, act_std

    @property
    def size(self):
        return self._size


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
            index = np.random.randint(
                self._bottom, self._bottom + self._size) % self._max_pool_size
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


class DDPG(RLAlgorithm):
    """
    Deep Deterministic Policy Gradient.
    """

    def __init__(
            self,
            env,
            policy,
            qf,
            es,
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
            soft_target=True,
            soft_target_tau=0.001,
            n_updates_per_sample=1,
            scale_reward=1.0,
            include_horizon_terminal_transitions=False,
            plot=False,
            # exploration params
            eta=1.,
            snn_n_samples=10,
            prior_sd=0.5,
            use_kl_ratio=False,
            kl_q_len=10,
            use_reverse_kl_reg=False,
            reverse_kl_reg_factor=1e-3,
            use_replay_pool=True,
            dyn_replay_pool_size=100000,
            dyn_min_pool_size=500,
            dyn_n_updates_per_sample=10,
            pool_batch_size=10,
            eta_discount=1.0,
            n_itr_update=5,
            reward_alpha=0.001,
            kl_alpha=0.001,
            normalize_reward=False,
            kl_batch_size=1,
            use_kl_ratio_q=False,
            unn_n_hidden=[32],
            unn_layers_type=[1, 1],
            unn_learning_rate=0.001,
            second_order_update=False,
            dyn_replay_freq=100,
            reset_expl_policy_freq=1000,
            exploration=True,
            compression=False,
            information_gain=True,
            vime=True,
    ):
        """
        :param env: Environment
        :param policy: Policy
        :param qf: Q function
        :param es: Exploration strategy
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
        :param soft_target_tau: Interpolation parameter for doing the soft target update.
        :param n_updates_per_sample: Number of Q function and policy updates per new sample obtained
        :param scale_reward: The scaling factor applied to the rewards when training
        :param include_horizon_terminal_transitions: whether to include transitions with terminal=True because the
        horizon was reached. This might make the Q value back up less stable for certain tasks.
        :param plot: Whether to visualize the policy performance after each eval_interval.
        :return:
        """
        self.env = env
        self.policy = policy
        self.qf = qf
        self.es = es
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
        self.soft_target_tau = soft_target_tau
        self.n_updates_per_sample = n_updates_per_sample
        self.include_horizon_terminal_transitions = include_horizon_terminal_transitions
        self.plot = plot

        self.qf_loss_averages = []
        self.policy_surr_averages = []
        self.q_averages = []
        self.y_averages = []
        self.paths = []
        self.es_path_returns = []
        self.paths_samples_cnt = 0

        self.scale_reward = scale_reward

        self.opt_info = None
        self.expl_opt_info = None

        # Set exploration params
        # ----------------------
        self.eta = eta
        self.snn_n_samples = snn_n_samples
        self.prior_sd = prior_sd
        self.use_kl_ratio = use_kl_ratio
        self.kl_q_len = kl_q_len
        self.use_reverse_kl_reg = use_reverse_kl_reg
        self.reverse_kl_reg_factor = reverse_kl_reg_factor
        self.use_replay_pool = use_replay_pool
        self.dyn_replay_pool_size = dyn_replay_pool_size
        self.dyn_min_pool_size = dyn_min_pool_size
        self.dyn_n_updates_per_sample = dyn_n_updates_per_sample
        self.pool_batch_size = pool_batch_size
        self.eta_discount = eta_discount
        self.n_itr_update = n_itr_update
        self.reward_alpha = reward_alpha
        self.kl_alpha = kl_alpha
        self.normalize_reward = normalize_reward
        self.kl_batch_size = kl_batch_size
        self.use_kl_ratio_q = use_kl_ratio_q
        self.unn_n_hidden = unn_n_hidden
        self.unn_layers_type = unn_layers_type
        self.unn_learning_rate = unn_learning_rate
        self.second_order_update = second_order_update
        self.dyn_replay_freq = dyn_replay_freq
        self.reset_expl_policy_freq = reset_expl_policy_freq
        self.expl_policy = pickle.loads(pickle.dumps(self.policy))
        self.expl_qf = pickle.loads(pickle.dumps(self.qf))
        self.compression = compression
        self.information_gain = information_gain
        self.vime = vime
        # ----------------------

        # Params to keep track of moving average (both intrinsic and external
        # reward) mean/var.
        if self.normalize_reward:
            self._reward_mean = deque(maxlen=self.kl_q_len)
            self._reward_std = deque(maxlen=self.kl_q_len)
        if self.use_kl_ratio:
            self._kl_mean = deque(maxlen=self.kl_q_len)
            self._kl_std = deque(maxlen=self.kl_q_len)

        if self.use_kl_ratio_q:
            # Add Queue here to keep track of N last kl values, compute average
            # over them and divide current kl values by it. This counters the
            # exploding kl value problem.
            self.kl_previous = deque(maxlen=self.kl_q_len)

    def start_worker(self):
        parallel_sampler.populate_task(self.env, self.policy)
        if self.plot:
            plotter.init_plot(self.env, self.policy)

    @overrides
    def train(self):

        # Uncertain neural network (UNN) initialization.
        # ------------------------------------------------
        batch_size = 1
        n_batches = 5  # FIXME, there is no correct value!

        # MDP observation and action dimensions.
        obs_dim = np.sum(self.env.observation_space.shape)
        act_dim = np.sum(self.env.action_space.shape)

        logger.log("Building UNN model (eta={}) ...".format(self.eta))
        start_time = time.time()

        self.vbnn = vbnn.VBNN(
            n_in=(obs_dim + act_dim),
            n_hidden=self.unn_n_hidden,
            n_out=obs_dim,
            n_batches=n_batches,
            layers_type=self.unn_layers_type,
            trans_func=lasagne.nonlinearities.rectify,
            out_func=lasagne.nonlinearities.linear,
            batch_size=batch_size,
            n_samples=self.snn_n_samples,
            prior_sd=self.prior_sd,
            use_reverse_kl_reg=self.use_reverse_kl_reg,
            reverse_kl_reg_factor=self.reverse_kl_reg_factor,
            #             stochastic_output=self.stochastic_output,
            second_order_update=self.second_order_update,
            learning_rate=self.unn_learning_rate,
            compression=self.compression,
            information_gain=self.information_gain
        )

        logger.log(
            "Model built ({:.1f} sec).".format((time.time() - start_time)))

        if self.use_replay_pool:
            dyn_pool = DynamicsSimpleReplayPool(
                max_pool_size=self.dyn_replay_pool_size,
                observation_shape=self.env.observation_space.shape,
                action_dim=act_dim
            )
        # ------------------------------------------------

        # This seems like a rather sequential method
        pool = SimpleReplayPool(
            max_pool_size=self.replay_pool_size,
            observation_dim=self.env.observation_space.flat_dim,
            action_dim=self.env.action_space.flat_dim,
        )
        self.start_worker()

        self.init_opt()
        self.init_opt_exploration()
        itr = 0
        path_length = 0
        path_return = 0
        terminal = False
        observation = self.env.reset()

        for epoch in xrange(self.n_epochs):
            logger.push_prefix('epoch #%d | ' % epoch)
            logger.log("Training started")

            kls = []
            for epoch_itr in pyprind.prog_bar(xrange(self.epoch_length)):
                # Execute policy
                if terminal:  # or path_length > self.max_path_length:
                    # Note that if the last time step ends an episode, the very
                    # last state and observation will be ignored and not added
                    # to the replay pool
                    observation = self.env.reset()
                    self.es.reset()
                    self.expl_policy.reset()
                    self.es_path_returns.append(path_return)
                    path_length = 0
                    path_return = 0

                action = self.es.get_action(
                    itr, observation, policy=self.expl_policy)  # qf=qf

                next_observation, reward, terminal, _ = self.env.step(action)
                path_length += 1
                path_return += reward

                if not terminal and path_length >= self.max_path_length:
                    terminal = True
                    # only include the terminal transition in this case if the
                    # flag was set
                    if self.include_horizon_terminal_transitions:
                        pool.add_sample(
                            observation, action, reward * self.scale_reward, terminal)
                    if self.use_replay_pool:
                        dyn_pool.add_sample(
                            observation, action, reward, terminal)
                else:
                    pool.add_sample(
                        observation, action, reward * self.scale_reward, terminal)
                    if self.use_replay_pool:
                        dyn_pool.add_sample(
                            observation, action, reward, terminal)

                if self.vime:
                    # Training dynamics model
                    # -----------------------
                    if itr % self.dyn_replay_freq == 0:
                        if self.use_replay_pool:
                            # Now we train the dynamics model using the replay dyn_pool; only
                            # if dyn_pool is large enough.
                            if dyn_pool.size >= self.dyn_min_pool_size:
                                _inputss = []
                                _targetss = []
                                for _ in xrange(self.dyn_n_updates_per_sample):
                                    batch = dyn_pool.random_batch(
                                        self.pool_batch_size)
                                    obs = batch['observations']
                                    next_obs = batch['next_observations']
                                    act = batch['actions']
                                    _inputs = np.hstack(
                                        [obs, act])
                                    _targets = next_obs
                                    _inputss.append(_inputs)
                                    _targetss.append(_targets)

                                for _inputs, _targets in zip(_inputss, _targetss):
                                    self.vbnn.train_fn(_inputs, _targets)
                    # -----------------------

                # Update next observation
                observation = next_observation

                if pool.size >= self.min_pool_size:
                    # Here we train actual policy.
                    for update_itr in xrange(self.n_updates_per_sample):
                        # Train policy
                        batch = pool.random_batch(self.batch_size)
                        self.do_training(itr, batch)

                    if self.vime:
                        # Every n iterations, set sample policy to match actual
                        # policy
                        if itr % self.reset_expl_policy_freq == 0:
                            logger.log(
                                'Copying policy params over to exploration policy.')
                            self.expl_policy.set_param_values(
                                self.policy.get_param_values())

                        for update_itr in xrange(self.n_updates_per_sample):
                            batch = pool.random_batch(self.batch_size)

                            # Calculate intrinsic rewards.
                            # ----------------------------
                            # Iterate over all paths and compute intrinsic reward by updating the
                            # model on each observation, calculating the KL divergence of the new
                            # params to the old ones, and undoing this
                            # operation.
                            obs = batch['observations']
                            act = batch['actions']
                            rew = batch['rewards']
                            obs_next = batch['next_observations']

                            # inputs = (o,a), target = o'
                            _inputs = np.hstack((obs, act))
                            _targets = obs_next

                            # KL vector assumes same shape as reward.
                            kl = np.zeros(rew.shape)

                            for j in xrange(obs.shape[0]):

                                # Save old params for every update.
                                self.vbnn.save_old_params()

                                start = j
                                end = np.minimum(
                                    (j + 1), obs.shape[0] - 1)

                                if self.second_order_update:
                                    # We do a line search over the best step sizes using
                                    # step_size * invH * grad
                                    best_loss_value = np.inf
                                    for step_size in [0.01]:
                                        self.vbnn.save_old_params()
                                        loss_value = self.vbnn.train_update_fn(
                                            _inputs[start:end], _targets[start:end], step_size)
                                        if loss_value < best_loss_value:
                                            best_loss_value = loss_value
                                        kl_div = np.clip(
                                            float(self.vbnn.f_kl_div_closed_form()), 0, 1000)
                                        # If using replay pool, undo updates.
                                        if self.use_replay_pool:
                                            self.vbnn.reset_to_old_params()
                                else:
                                    # Update model weights based on current
                                    # minibatch.
                                    for _ in xrange(self.n_itr_update):
                                        self.vbnn.train_update_fn(
                                            _inputs[start:end], _targets[start:end])

                                    # Calculate current minibatch KL.
                                    kl_div = np.clip(
                                        float(self.vbnn.f_kl_div_closed_form()), 0, 1000)

                                for k in xrange(start, end):
                                    kl[k] = kl_div

                                # If using replay pool, undo updates.
                                if self.use_replay_pool:
                                    self.vbnn.reset_to_old_params()

                            # Store original KL values for averaging through kl
                            # ratios.
                            kls.append(kl)

                            # Perform normlization of the intrinsic rewards.
                            if self.use_kl_ratio and self.use_kl_ratio_q:
                                # Update kl Q
                                if len(self.kl_previous) > 0:
                                    previous_mean_kl = np.median(
                                        np.asarray(self.kl_previous))
                                    # Add KL ass intrinsic reward to external
                                    # reward
                                    batch['rewards'] = batch['rewards'] + \
                                        self.eta * kl / \
                                        previous_mean_kl  # * np.mean(self.q_averages[-1000:])
                            else:
                                # Add KL ass intrinsic reward to external
                                # reward
                                batch['rewards'] = batch[
                                    'rewards'] + self.eta * kl
                            # ----------------------------

                        # Train exploration policy
                        self.do_training_exploration(itr, batch)
                    else:
                        self.expl_policy.set_param_values(
                            self.policy.get_param_values())

                itr += 1

            if len(kls) != 0 and self.use_kl_ratio and self.use_kl_ratio_q:
                # Update kl Q at the end of each epoch.
                self.kl_previous.append(np.mean(np.hstack(kls)))

            # Discount eta at the end of each epoch.
            self.eta *= self.eta_discount

            logger.log("Training finished")
            if pool.size >= self.min_pool_size:
                self.evaluate(epoch, pool)
                params = self.get_epoch_snapshot(epoch)
                logger.save_itr_params(epoch, params)
                if self.vime:
                    logger.record_tabular(
                        'VBNN_MeanKL', np.mean(np.hstack(kls)))
                    logger.record_tabular('VBNN_MedianKL',
                                          np.median(np.hstack(kls)))
                    logger.record_tabular('VBNN_StdKL', np.std(np.hstack(kls)))
                    logger.record_tabular('VBNN_MinKL', np.min(np.hstack(kls)))
                    logger.record_tabular('VBNN_MaxKL', np.max(np.hstack(kls)))
            logger.dump_tabular(with_prefix=False)
            logger.pop_prefix()

    def init_opt(self):

        # First, create "target" policy and Q functions
        target_policy = pickle.loads(pickle.dumps(self.policy))
        target_qf = pickle.loads(pickle.dumps(self.qf))

        # y need to be computed first
        obs = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1,
        )

        # The yi values are computed separately as above and then passed to
        # the training functions below
        action = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1,
        )
        yvar = TT.vector('ys')

        qf_weight_decay_term = 0.5 * self.qf_weight_decay * \
            sum([TT.sum(TT.square(param)) for param in
                 self.qf.get_params(regularizable=True)])

        qval = self.qf.get_qval_sym(obs, action)

        qf_loss = TT.mean(TT.square(yvar - qval))
        qf_reg_loss = qf_loss + qf_weight_decay_term

        policy_weight_decay_term = 0.5 * self.policy_weight_decay * \
            sum([TT.sum(TT.square(param))
                 for param in self.policy.get_params(regularizable=True)])
        policy_qval = self.qf.get_qval_sym(
            obs, self.policy.get_action_sym(obs),
            deterministic=True
        )
        policy_surr = -TT.mean(policy_qval)

        policy_reg_surr = policy_surr + policy_weight_decay_term

        qf_updates = self.qf_update_method(
            qf_reg_loss, self.qf.get_params(trainable=True))
        policy_updates = self.policy_update_method(
            policy_reg_surr, self.policy.get_params(trainable=True))

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

        self.opt_info = dict(
            f_train_qf=f_train_qf,
            f_train_policy=f_train_policy,
            target_qf=target_qf,
            target_policy=target_policy,
        )

    def init_opt_exploration(self):

        # First, create "target" policy and Q functions
        target_policy = pickle.loads(pickle.dumps(self.expl_policy))
        target_qf = pickle.loads(pickle.dumps(self.expl_qf))

        # y need to be computed first
        obs = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1,
        )

        # The yi values are computed separately as above and then passed to
        # the training functions below
        action = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1,
        )
        yvar = TT.vector('ys')

        qf_weight_decay_term = 0.5 * self.qf_weight_decay * \
            sum([TT.sum(TT.square(param)) for param in
                 self.expl_qf.get_params(regularizable=True)])

        qval = self.expl_qf.get_qval_sym(obs, action)

        qf_loss = TT.mean(TT.square(yvar - qval))
        qf_reg_loss = qf_loss + qf_weight_decay_term

        policy_weight_decay_term = 0.5 * self.policy_weight_decay * \
            sum([TT.sum(TT.square(param))
                 for param in self.expl_policy.get_params(regularizable=True)])
        policy_qval = self.expl_qf.get_qval_sym(
            obs, self.expl_policy.get_action_sym(obs),
            deterministic=True
        )
        policy_surr = -TT.mean(policy_qval)

        policy_reg_surr = policy_surr + policy_weight_decay_term

        qf_updates = self.qf_update_method(
            qf_reg_loss, self.expl_qf.get_params(trainable=True))
        policy_updates = self.policy_update_method(
            policy_reg_surr, self.expl_policy.get_params(trainable=True))

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

        self.expl_opt_info = dict(
            f_train_qf=f_train_qf,
            f_train_policy=f_train_policy,
            target_qf=target_qf,
            target_policy=target_policy,
        )

    def do_training_exploration(self, itr, batch):

        obs, actions, rewards, next_obs, terminals = ext.extract(
            batch,
            "observations", "actions", "rewards", "next_observations",
            "terminals"
        )

        # compute the on-policy y values
        target_qf = self.expl_opt_info["target_qf"]
        target_policy = self.expl_opt_info["target_policy"]

        next_actions, _ = target_policy.get_actions(next_obs)
        next_qvals = target_qf.get_qval(next_obs, next_actions)

        ys = rewards + (1. - terminals) * self.discount * next_qvals

        f_train_qf = self.expl_opt_info["f_train_qf"]
        f_train_policy = self.expl_opt_info["f_train_policy"]

        f_train_qf(ys, obs, actions)

        f_train_policy(obs)

        target_policy.set_param_values(
            target_policy.get_param_values() * (1.0 - self.soft_target_tau) +
            self.expl_policy.get_param_values() * self.soft_target_tau)
        target_qf.set_param_values(
            target_qf.get_param_values() * (1.0 - self.soft_target_tau) +
            self.expl_qf.get_param_values() * self.soft_target_tau)

    def do_training(self, itr, batch):

        obs, actions, rewards, next_obs, terminals = ext.extract(
            batch,
            "observations", "actions", "rewards", "next_observations",
            "terminals"
        )

        # compute the on-policy y values
        target_qf = self.opt_info["target_qf"]
        target_policy = self.opt_info["target_policy"]

        next_actions, _ = target_policy.get_actions(next_obs)
        next_qvals = target_qf.get_qval(next_obs, next_actions)

        ys = rewards + (1. - terminals) * self.discount * next_qvals

        f_train_qf = self.opt_info["f_train_qf"]
        f_train_policy = self.opt_info["f_train_policy"]

        qf_loss, qval = f_train_qf(ys, obs, actions)

        policy_surr = f_train_policy(obs)

        target_policy.set_param_values(
            target_policy.get_param_values() * (1.0 - self.soft_target_tau) +
            self.policy.get_param_values() * self.soft_target_tau)
        target_qf.set_param_values(
            target_qf.get_param_values() * (1.0 - self.soft_target_tau) +
            self.qf.get_param_values() * self.soft_target_tau)

        self.qf_loss_averages.append(qf_loss)
        self.policy_surr_averages.append(policy_surr)
        self.q_averages.append(qval)
        self.y_averages.append(ys)

    def evaluate(self, epoch, pool):
        logger.log("Collecting samples for evaluation")
        paths = parallel_sampler.sample_paths(
            policy_params=self.policy.get_param_values(),
            max_samples=self.eval_samples,
            max_path_length=self.max_path_length,
        )

        average_discounted_return = np.mean(
            [special.discount_return(
                path["rewards"], self.discount) for path in paths]
        )

        returns = [sum(path["rewards"]) for path in paths]

        all_qs = np.concatenate(self.q_averages)
        all_ys = np.concatenate(self.y_averages)

        average_q_loss = np.mean(self.qf_loss_averages)
        average_policy_surr = np.mean(self.policy_surr_averages)
        average_action = np.mean(np.square(np.concatenate(
            [path["actions"] for path in paths]
        )))

        policy_reg_param_norm = np.linalg.norm(
            self.policy.get_param_values(regularizable=True)
        )
        qfun_reg_param_norm = np.linalg.norm(
            self.qf.get_param_values(regularizable=True)
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
        logger.record_tabular('AverageDiscountedReturn',
                              average_discounted_return)
        if len(self.es_path_returns) > 0:
            logger.record_tabular('AverageEsReturn',
                                  np.mean(self.es_path_returns))
            logger.record_tabular('StdEsReturn',
                                  np.std(self.es_path_returns))
            logger.record_tabular('MaxEsReturn',
                                  np.max(self.es_path_returns))
            logger.record_tabular('MinEsReturn',
                                  np.min(self.es_path_returns))
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

        self.env.log_diagnostics(paths)
        self.policy.log_diagnostics(paths)

        self.qf_loss_averages = []
        self.policy_surr_averages = []

        self.q_averages = []
        self.y_averages = []
        self.es_path_returns = []

    def get_epoch_snapshot(self, epoch):
        return dict(
            env=self.env,
            epoch=epoch,
            qf=self.qf,
            policy=self.policy,
            target_qf=self.opt_info["target_qf"],
            target_policy=self.opt_info["target_policy"],
            es=self.es,
        )
