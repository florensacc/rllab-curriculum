from rllab.algos.base import RLAlgorithm

import numpy as np

from rllab.misc.special import discount_cumsum
from rllab.sampler import parallel_sampler, stateful_pool
from rllab.sampler.utils import rollout
from rllab.core.serializable import Serializable
import rllab.misc.logger as logger
import rllab.plotter as plotter

# exploration imports
# -------------------
import theano
import lasagne
from collections import deque
import time
from sandbox.rein.dynamics_models.nn_uncertainty import vbnn
# -------------------


class SimpleReplayPool(object):
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


def _worker_rollout_policy(G, args):
    sample_std = args["sample_std"].flatten()
    cur_mean = args["cur_mean"].flatten()
    K = len(cur_mean)
    params = np.random.standard_normal(K) * sample_std + cur_mean
    G.policy.set_param_values(params)
    path = rollout(G.env, G.policy, args["max_path_length"])

    # Original reward.
    path['rewards_orig'] = np.array(path['rewards'])
    # Returns are computed later on
    path['returns'] = np.nan
    
    path["undiscounted_return"] = sum(path["rewards_orig"])
    if args["criterion"] == "samples":
        inc = len(path["rewards"])
    elif args["criterion"] == "paths":
        inc = 1
    else:
        raise NotImplementedError
    return (params, path), inc


class CEM(RLAlgorithm, Serializable):

    def __init__(
            self,
            env,
            policy,
            n_itr=500,
            max_path_length=500,
            discount=0.99,
            init_std=1.,
            n_samples=100,
            batch_size=None,
            best_frac=0.05,
            extra_std=1.,
            extra_decay_time=100,
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
            replay_pool_size=100000,
            min_pool_size=500,
            n_updates_per_sample=500,
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
            compression=False,
            information_gain=True,
            **kwargs
    ):
        """
        :param n_itr: Number of iterations.
        :param max_path_length: Maximum length of a single rollout.
        :param batch_size: # of samples from trajs from param distribution, when this
        is set, n_samples is ignored
        :param discount: Discount.
        :param plot: Plot evaluation run after each iteration.
        :param init_std: Initial std for param distribution
        :param extra_std: Decaying std added to param distribution at each iteration
        :param extra_decay_time: Iterations that it takes to decay extra std
        :param n_samples: #of samples from param distribution
        :param best_frac: Best fraction of the sampled params
        :return:
        """
        Serializable.quick_init(self, locals())
        self.env = env
        self.policy = policy
        self.batch_size = batch_size
        self.plot = plot
        self.extra_decay_time = extra_decay_time
        self.extra_std = extra_std
        self.best_frac = best_frac
        self.n_samples = n_samples
        self.init_std = init_std
        self.discount = discount
        self.max_path_length = max_path_length
        self.n_itr = n_itr

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
        self.replay_pool_size = replay_pool_size
        self.min_pool_size = min_pool_size
        self.n_updates_per_sample = n_updates_per_sample
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
        self.compression = compression
        self.information_gain = information_gain
        # ----------------------

        if self.second_order_update:
            assert self.kl_batch_size == 1
            assert self.n_itr_update == 1

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
            self.pool = SimpleReplayPool(
                max_pool_size=self.replay_pool_size,
                observation_shape=self.env.observation_space.shape,
                action_dim=act_dim
            )
        # ------------------------------------------------

        parallel_sampler.populate_task(self.env, self.policy)
        if self.plot:
            plotter.init_plot(self.env, self.policy)

        cur_std = self.init_std
        cur_mean = self.policy.get_param_values()
        # K = cur_mean.size
        n_best = max(1, int(self.n_samples * self.best_frac))

        for itr in range(self.n_itr):
            # sample around the current distribution
            extra_var_mult = max(1.0 - itr / self.extra_decay_time, 0)
            sample_std = np.sqrt(
                np.square(cur_std) + np.square(self.extra_std) * extra_var_mult)
            if self.batch_size is None:
                criterion = 'paths'
                threshold = self.n_samples
            else:
                criterion = 'samples'
                threshold = self.batch_size

            infos = stateful_pool.singleton_pool.run_collect(
                _worker_rollout_policy,
                threshold=threshold,
                args=(dict(cur_mean=cur_mean,
                           sample_std=sample_std,
                           max_path_length=self.max_path_length,
                           discount=self.discount,
                           criterion=criterion),)
            )
            xs = np.asarray([info[0] for info in infos])
            paths = [info[1] for info in infos]

            if self.normalize_reward:
                # Update reward mean/std Q.
                rewards = []
                for i in xrange(len(paths)):
                    rewards.append(paths[i]['rewards'])
                rewards_flat = np.hstack(rewards)
                self._reward_mean.append(np.mean(rewards_flat))
                self._reward_std.append(np.std(rewards_flat))

                # Normalize rewards.
                reward_mean = np.mean(np.asarray(self._reward_mean))
                reward_std = np.mean(np.asarray(self._reward_std))
                for i in xrange(len(paths)):
                    paths[i]['rewards'] = (
                        paths[i]['rewards'] - reward_mean) / (reward_std + 1e-8)

            # Computing intrinsic rewards.
            # ----------------------------
            if itr > 0:
                logger.log("Computing intrinsic rewards ...")

                # List to keep track of kl per path.
                kls = []

                # Iterate over all paths and compute intrinsic reward by updating the
                # model on each observation, calculating the KL divergence of the new
                # params to the old ones, and undoing this operation.
                obs_mean, obs_std, act_mean, act_std = self.pool.mean_obs_act()
                for i in xrange(len(paths)):
                    obs = (paths[i]['observations'] - obs_mean) / \
                        (obs_std + 1e-8)
                    act = (paths[i]['actions'] - act_mean) / (act_std + 1e-8)
                    rew = paths[i]['rewards']

                    # inputs = (o,a), target = o'
                    obs_nxt = np.vstack([obs[1:]])
                    _inputs = np.hstack([obs[:-1], act[:-1]])
                    _targets = obs_nxt

                    # KL vector assumes same shape as reward.
                    kl = np.zeros(rew.shape)

                    for j in xrange(int(np.ceil(obs.shape[0] / float(self.kl_batch_size)))):

                        # Save old params for every update.
                        self.vbnn.save_old_params()

                        start = j * self.kl_batch_size
                        end = np.minimum(
                            (j + 1) * self.kl_batch_size, obs.shape[0] - 1)

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
                                    float(self.vbnn.f_kl_div_closed_form()), 0, 50)
                                # If using replay pool, undo updates.
                                if self.use_replay_pool:
                                    self.vbnn.reset_to_old_params()
                        else:
                            # Update model weights based on current minibatch.
                            for _ in xrange(self.n_itr_update):
                                self.vbnn.train_update_fn(
                                    _inputs[start:end], _targets[start:end])

                            # Calculate current minibatch KL.
                            kl_div = np.clip(
                                float(self.vbnn.f_kl_div_closed_form()), 0, 50)

                        for k in xrange(start, end):
                            kl[k] = kl_div

                        # If using replay pool, undo updates.
                        if self.use_replay_pool:
                            self.vbnn.reset_to_old_params()

                    # Last element in KL vector needs to be replaced by second last one
                    # because the actual last observation has no next
                    # observation.
                    kl[-1] = kl[-2]
                    # Add kl to list of kls
                    kls.append(kl)

                kls_flat = np.hstack(kls)

                # Perform normlization of the intrinsic rewards.
                if self.use_kl_ratio:
                    if self.use_kl_ratio_q:
                        # Update kl Q
                        self.kl_previous.append(np.median(np.hstack(kls)))
                        previous_mean_kl = np.mean(
                            np.asarray(self.kl_previous))
                        for i in xrange(len(kls)):
                            kls[i] = kls[i] / previous_mean_kl

                # Add KL ass intrinsic reward to external reward
                for i in xrange(len(paths)):
                    paths[i]['rewards'] = paths[i][
                        'rewards'] + self.eta * kls[i]

                # Discount eta
                self.eta *= self.eta_discount

                logger.log("Intrinsic reward computed.")
                # ----------------------------

            # Compute discounted returns
            for i in xrange(len(paths)):
                paths[i]["returns"] = discount_cumsum(
                    paths[i]["rewards"], self.discount)

            # Exploration code
            # ----------------
            if self.use_replay_pool:
                # Fill replay pool.
                logger.log("Fitting dynamics model using replay pool ...")
                for path in paths:
                    path_len = len(path['rewards'])
                    for i in xrange(path_len):
                        obs = path['observations'][i]
                        act = path['actions'][i]
                        rew = path['rewards'][i]
                        term = (i == path_len - 1)
                        self.pool.add_sample(obs, act, rew, term)

                # Now we train the dynamics model using the replay self.pool; only
                # if self.pool is large enough.
                if self.pool.size >= self.min_pool_size:
                    obs_mean, obs_std, act_mean, act_std = self.pool.mean_obs_act()
                    _inputss = []
                    _targetss = []
                    for _ in xrange(self.n_updates_per_sample):
                        batch = self.pool.random_batch(
                            self.pool_batch_size)
                        obs = (batch['observations'] - obs_mean) / \
                            (obs_std + 1e-8)
                        next_obs = (
                            batch['next_observations'] - obs_mean) / (obs_std + 1e-8)
                        act = (batch['actions'] - act_mean) / \
                            (act_std + 1e-8)
                        _inputs = np.hstack(
                            [obs, act])
                        _targets = next_obs
                        _inputss.append(_inputs)
                        _targetss.append(_targets)

                    old_acc = 0.
                    for _inputs, _targets in zip(_inputss, _targetss):
                        _out = self.vbnn.pred_fn(_inputs)
                        old_acc += np.mean(np.square(_out - _targets))
                    old_acc /= len(_inputss)

                    for _inputs, _targets in zip(_inputss, _targetss):
                        self.vbnn.train_fn(_inputs, _targets)

                    new_acc = 0.
                    for _inputs, _targets in zip(_inputss, _targetss):
                        _out = self.vbnn.pred_fn(_inputs)
                        new_acc += np.mean(np.square(_out - _targets))
                    new_acc /= len(_inputss)

                    logger.record_tabular(
                        'SNN_DynModelSqLossBefore', old_acc)
                    logger.record_tabular(
                        'SNN_DynModelSqLossAfter', new_acc)
                # ----------------

            fs = np.array([path['returns'][0] for path in paths])
            best_inds = (-fs).argsort()[:n_best]
            best_xs = xs[best_inds]
            cur_mean = best_xs.mean(axis=0)
            cur_std = best_xs.std(axis=0)
            best_x = best_xs[0]
            logger.push_prefix('itr #%d | ' % itr)
            logger.record_tabular('Iteration', itr)
            logger.record_tabular('CurStdMean', np.mean(cur_std))
            undiscounted_returns = np.array(
                [path['undiscounted_return'] for path in paths])
            logger.record_tabular('AverageReturn',
                                  np.mean(undiscounted_returns))
            logger.record_tabular('StdReturn',
                                  np.mean(undiscounted_returns))
            logger.record_tabular('MaxReturn',
                                  np.max(undiscounted_returns))
            logger.record_tabular('MinReturn',
                                  np.min(undiscounted_returns))
            logger.record_tabular('AverageDiscountedReturn',
                                  np.mean(fs))
            logger.record_tabular('AvgTrajLen',
                                  np.mean([len(path['returns']) for path in paths]))
            logger.record_tabular('NumTrajs',
                                  len(paths))
            # Exploration logged vars.
            # ------------------------
            if itr > 0:
                logger.record_tabular('VBNN_MedianKL', np.median(kls_flat))
                logger.record_tabular('VBNN_MeanKL', np.mean(kls_flat))
                logger.record_tabular('VBNN_StdKL', np.std(kls_flat))
                logger.record_tabular('VBNN_MinKL', np.min(kls_flat))
                logger.record_tabular('VBNN_MaxKL', np.max(kls_flat))
            else:
                logger.record_tabular('VBNN_MedianKL', 0.)
                logger.record_tabular('VBNN_MeanKL', 0.)
                logger.record_tabular('VBNN_StdKL', 0.)
                logger.record_tabular('VBNN_MinKL', 0.)
                logger.record_tabular('VBNN_MaxKL', 0.)
            self.policy.set_param_values(best_x)
            self.env.log_diagnostics(paths)
            self.policy.log_diagnostics(paths)
            logger.save_itr_params(itr, dict(
                itr=itr,
                policy=self.policy,
                env=self.env,
                cur_mean=cur_mean,
                cur_std=cur_std,
            ))
            logger.dump_tabular(with_prefix=False)
            logger.pop_prefix()
            if self.plot:
                plotter.update_plot(self.policy, self.max_path_length)
        parallel_sampler.terminate_task()
