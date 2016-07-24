import numpy as np

from rllab.algos.base import RLAlgorithm
from sandbox.rein.sampler import parallel_sampler_vime as parallel_sampler
from rllab.misc import special
from rllab.misc import tensor_utils
from rllab.algos import util
import rllab.misc.logger as logger
import rllab.plotter as plotter
from sandbox.rein.dynamics_models.utils import iterate_minibatches, group, ungroup,\
    plot_mnist_digit
from scipy import stats, misc
from sandbox.rein.dynamics_models.utils import enum, atari_format_image, atari_unformat_image
# Nonscientific printing of numpy arrays.
np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)

# exploration imports
# -------------------
import theano
import lasagne
from collections import deque
import time
from sandbox.rein.dynamics_models.bnn import conv_bnn_vime
# -------------------


class SimpleReplayPool(object):
    """Replay pool"""

    def __init__(
            self,
            max_pool_size,
            observation_shape,
            action_dim,
            observation_dtype=theano.config.floatX,
            action_dtype=theano.config.floatX,
    ):
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

    def __str__(self):
        sb = []
        for key in self.__dict__:
            sb.append(
                "{key}='{value}'".format(key=key, value=self.__dict__[key]))
        return ', '.join(sb)

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
        #         if self._size >= self._max_pool_size:
        #             obs = self._observations
        #             act = self._actions
        #         else:
        #             obs = self._observations[:self._top + 1]
        #             act = self._actions[:self._top + 1]
        #         obs_mean = np.mean(obs, axis=0)
        #         obs_std = np.std(obs, axis=0)
        #         act_mean = np.mean(act, axis=0)
        #         act_std = np.std(act, axis=0)
        #         return obs_mean, obs_std, act_mean, act_std
        return 0., 0., 1., 1.

    @property
    def size(self):
        return self._size


class BatchPolopt(RLAlgorithm):
    """
    Base class for batch sampling-based policy optimization methods.
    This includes various policy gradient methods like vpg, npg, ppo, trpo, etc.
    """

    # Enums
    SurpriseTransform = enum(
        CAP1000='cap at 1000', LOG='log(1+surprise)', ZERO100='0-100', CAP90PERC='cap at 90th percentile')

    def __init__(
            self,
            env,
            policy,
            baseline,
            n_itr=500,
            start_itr=0,
            batch_size=5000,
            max_path_length=500,
            discount=0.99,
            gae_lambda=1,
            plot=False,
            pause_for_plot=False,
            whole_paths=True,
            center_adv=True,
            positive_adv=False,
            record_states=False,
            store_paths=False,
            algorithm_parallelized=False,
            # exploration params
            eta=1.,
            snn_n_samples=10,
            prior_sd=0.5,
            use_kl_ratio=False,
            kl_q_len=10,
            num_sample_updates=1,
            n_itr_update=5,
            reward_alpha=0.001,
            kl_alpha=0.001,
            normalize_reward=False,
            kl_batch_size=1,
            use_kl_ratio_q=False,
            layers_disc=None,
            state_dim=None,
            action_dim=None,
            reward_dim=None,
            unn_learning_rate=0.001,
            second_order_update=False,
            surprise_type='information_gain',
            surprise_transform=None,
            update_likelihood_sd=False,
            replay_kl_schedule=1.0,
            output_type='regression',
            use_local_reparametrization_trick=True,
            predict_reward=False,
            group_variance_by='weight',
            likelihood_sd_init=1.0,
            disable_variance=False,
            predict_delta=False,
            dyn_pool_args=dict(enable=False, size=100000, min_size=10, batch_size=32),
            ** kwargs
    ):
        """
        :param env: Environment
        :param policy: Policy
        :param baseline: Baseline
        :param n_itr: Number of iterations.
        :param start_itr: Starting iteration.
        :param batch_size: Number of samples per iteration.
        :param max_path_length: Maximum length of a single rollout.
        :param discount: Discount.
        :param gae_lambda: Lambda used for generalized advantage estimation.
        :param plot: Plot evaluation run after each iteration.
        :param pause_for_plot: Whether to pause before contiuing when plotting.
        :param whole_paths: Make sure that the samples contain whole trajectories, even if the actual batch size is
        slightly larger than the specified batch_size.
        :param center_adv: Whether to rescale the advantages so that they have mean 0 and standard deviation 1.
        :param positive_adv: Whether to shift the advantages so that they are always positive. When used in
        conjunction with center_adv the advantages will be standardized before shifting.
        :param store_paths: Whether to save all paths data to the snapshot.
        :return:
        """

        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.plot = plot
        self.pause_for_plot = pause_for_plot
        self.whole_paths = whole_paths
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.store_paths = store_paths

        # Set exploration params
        # ----------------------
        self.eta = eta
        self.snn_n_samples = snn_n_samples
        self.prior_sd = prior_sd
        self.use_kl_ratio = use_kl_ratio
        self.kl_q_len = kl_q_len
        self.num_sample_updates = num_sample_updates
        self.n_itr_update = n_itr_update
        self.reward_alpha = reward_alpha
        self.kl_alpha = kl_alpha
        self.normalize_reward = normalize_reward
        self.kl_batch_size = kl_batch_size
        self.use_kl_ratio_q = use_kl_ratio_q
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.layers_disc = layers_disc
        self.unn_learning_rate = unn_learning_rate
        self.second_order_update = second_order_update
        self.surprise_type = surprise_type
        self.surprise_transform = surprise_transform
        self.update_likelihood_sd = update_likelihood_sd
        self.replay_kl_schedule = replay_kl_schedule
        self.output_type = output_type
        self.use_local_reparametrization_trick = use_local_reparametrization_trick
        self.predict_reward = predict_reward
        self.group_variance_by = group_variance_by
        self.likelihood_sd_init = likelihood_sd_init
        self.disable_variance = disable_variance
        self._predict_delta = predict_delta
        self._dyn_pool_args = dyn_pool_args
        # ----------------------

        if self.surprise_type == conv_bnn_vime.ConvBNNVIME.SurpriseType.COMPR:
            assert self._dyn_pool_args['enable'] is False
            assert self.use_kl_ratio is False
            print('ATTENTION: running {} with second_order_update={}'.format(
                self.surprise_type, self.second_order_update))

        if self.second_order_update:
            assert self.n_itr_update == 1

        # Params to keep track of moving average (both intrinsic and external
        # reward) mean/var.
        if self.normalize_reward:
            self._reward_mean = deque(maxlen=self.kl_q_len)
            self._reward_std = deque(maxlen=self.kl_q_len)
        if self.use_kl_ratio:
            self._kl_mean = deque(maxlen=self.kl_q_len)
            self._kl_std = deque(maxlen=self.kl_q_len)

        # If not Q, we use median of each batch, perhaps more stable? Because
        # network is only updated between batches, might work out well.
        if self.use_kl_ratio_q:
            # Add Queue here to keep track of N last kl values, compute average
            # over them and divide current kl values by it. This counters the
            # exploding kl value problem.
            self.kl_previous = deque(maxlen=self.kl_q_len)

    def start_worker(self):
        parallel_sampler.populate_task(self.env, self.policy, self.bnn)
        if self.plot:
            plotter.init_plot(self.env, self.policy)

    def shutdown_worker(self):
        pass

    def accuracy(self, _inputs, _targets):
        acc = 0.
        _out = self.bnn.pred_fn(_inputs)
        if self.output_type == 'regression':
            acc += np.mean(np.sum(np.square(_out - _targets), axis=1))
        elif self.output_type == 'classification':
            # FIXME: only for Atari
            _out2 = _out.reshape([-1, 256])
            _argm = np.argmax(_out2, axis=1)
            _argm2 = _argm.reshape([-1, 128])
            acc += np.sum(np.equal(_targets, _argm2)) / float(_targets.size)
        return acc

    def plot_pred_imgs(self, inputs, targets, itr, count):
        try:
            # This is specific to Atari.
            import matplotlib.pyplot as plt
            if not hasattr(self, '_fig'):
                self._fig = plt.figure()
                self._fig_1 = self._fig.add_subplot(141)
                plt.tick_params(axis='both', which='both', bottom='off', top='off',
                                labelbottom='off', right='off', left='off', labelleft='off')
                self._fig_2 = self._fig.add_subplot(142)
                plt.tick_params(axis='both', which='both', bottom='off', top='off',
                                labelbottom='off', right='off', left='off', labelleft='off')
                self._fig_3 = self._fig.add_subplot(143)
                plt.tick_params(axis='both', which='both', bottom='off', top='off',
                                labelbottom='off', right='off', left='off', labelleft='off')
                self._fig_4 = self._fig.add_subplot(144)
                plt.tick_params(axis='both', which='both', bottom='off', top='off',
                                labelbottom='off', right='off', left='off', labelleft='off')
                self._im1, self._im2, self._im3, self._im4 = None, None, None, None
            sanity_pred = self.bnn.pred_fn(inputs)
            input_im = inputs[:, :-self.env.spec.action_space.flat_dim]
            input_im = input_im[0, :].reshape(self.state_dim).transpose(1, 2, 0)[:, :, 0]
            sanity_pred_im = sanity_pred[0, :-1].reshape(self.state_dim).transpose(1, 2, 0)[:, :, 0]
            target_im = targets[0, :-1].reshape(self.state_dim).transpose(1, 2, 0)[:, :, 0]

            if self._predict_delta:
                sanity_pred_im += input_im
                target_im += input_im

            sanity_pred_im = sanity_pred_im * 256.
            sanity_pred_im = np.around(sanity_pred_im).astype(int)
            target_im = target_im * 256.
            target_im = np.around(target_im).astype(int)
            err = np.abs(target_im - sanity_pred_im)
            input_im = input_im * 256.
            input_im = np.around(input_im).astype(int)

            if self._im1 is None or self._im2 is None:
                self._im1 = self._fig_1.imshow(
                    input_im, interpolation='none', cmap='Greys_r', vmin=0, vmax=255)
                self._im2 = self._fig_2.imshow(
                    target_im, interpolation='none', cmap='Greys_r', vmin=0, vmax=255)
                self._im3 = self._fig_3.imshow(
                    sanity_pred_im, interpolation='none', cmap='Greys_r', vmin=0, vmax=255)
                self._im4 = self._fig_4.imshow(
                    err, interpolation='none', cmap='Greys_r', vmin=0, vmax=255)
            else:
                self._im1.set_data(input_im)
                self._im2.set_data(target_im)
                self._im3.set_data(sanity_pred_im)
                self._im4.set_data(err)
            plt.savefig(
                logger._snapshot_dir + '/dynpred_img_{}_{}.png'.format(itr, count), bbox_inches='tight')
        except Exception:
            pass

    def train(self):

        # If we don't use a replay pool, we could have correct values here, as
        # it is purely Bayesian. We then divide the KL divergence term by the
        # number of batches in each iteration `batch'. Also the batch size
        # would be given correctly.
        if self._dyn_pool_args['enable']:
            logger.log('Dynamics replay pool is ON.')
            batch_size = 1
            n_batches = 50  # FIXME, there is no correct value!
        else:
            logger.log('Dynamics replay pool is OFF.')
            batch_size = 1
            n_batches = 1

        logger.log("Building BNN model (eta={}) ...".format(self.eta))
        start_time = time.time()

        self.bnn = conv_bnn_vime.ConvBNNVIME(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            reward_dim=self.reward_dim,
            layers_disc=self.layers_disc,
            n_batches=n_batches,
            trans_func=lasagne.nonlinearities.rectify,
            out_func=lasagne.nonlinearities.linear,
            batch_size=batch_size,
            n_samples=self.snn_n_samples,
            prior_sd=self.prior_sd,
            second_order_update=self.second_order_update,
            learning_rate=self.unn_learning_rate,
            surprise_type=self.surprise_type,
            update_prior=(not self._dyn_pool_args['enable']),
            update_likelihood_sd=self.update_likelihood_sd,
            group_variance_by=self.group_variance_by,
            output_type=self.output_type,
            num_classes=256,
            num_output_dim=128,
            use_local_reparametrization_trick=self.use_local_reparametrization_trick,
            likelihood_sd_init=self.likelihood_sd_init,
            disable_variance=self.disable_variance
        )

        # Number of weights in BNN, excluding biases.
        self.num_weights = self.bnn.num_weights()

        logger.log("Model built ({:.1f} sec, {} weights).".format((time.time() - start_time),
                                                                  self.num_weights))

        if self._dyn_pool_args['enable']:
            if self.output_type == conv_bnn_vime.ConvBNNVIME.OutputType.CLASSIFICATION:
                observation_dtype = int
            elif self.output_type == conv_bnn_vime.ConvBNNVIME.OutputType.REGRESSION:
                observation_dtype = float
            self.pool = SimpleReplayPool(
                max_pool_size=self.replay_pool_size,
                # self.env.observation_space.shape,
                observation_shape=(self.env.observation_space.flat_dim,),
                action_dim=self.env.action_dim,
                observation_dtype=observation_dtype,
                **self._pool_args
            )

        self.start_worker()
        self.init_opt()
        episode_rewards, episode_lengths = [], []
        acc_before, acc_after, train_loss = 0., 0., 0.

        # KL rescaling factor for replay pool-based training.
        kl_factor = 1.0

        # ATTENTION: important to know when 'rewards' and 'rewards_orig' needs
        # to be used!
        for itr in xrange(self.start_itr, self.n_itr):
            logger.push_prefix('itr #%d | ' % itr)

            paths = self.obtain_samples(itr)

            if self._dyn_pool_args['enable']:
                # USE REPLAY POOL
                # ---------------
                # Fill replay pool with samples of current batch. Exclude the
                # last one.
                logger.log("Fitting dynamics model using replay pool ...")
                for path in paths:
                    path_len = len(path['rewards'])
                    for i in xrange(path_len):
                        obs = path['observations'][i]
                        act = path['actions'][i]
                        rew_orig = path['rewards_orig'][i]
                        term = (i == path_len - 1)
                        if not term:
                            self.pool.add_sample(obs, act, rew_orig, term)

                # Now we train the dynamics model using the replay self.pool; only
                # if self.pool is large enough.
                if self.pool.size >= self.min_pool_size:
                    obs_mean, obs_std, act_mean, act_std = self.pool.mean_obs_act()
                    acc_before, acc_after, train_loss = 0., 0., 0.
                    itr_tot = int(
                        np.ceil(self.num_sample_updates * float(self.batch_size) / self._dyn_pool_args['batch_size']))

                    for i in xrange(itr_tot):

                        batch = self.pool.random_batch(self._dyn_pool_args['batch_size'])

                        act = (batch['actions'] - act_mean) / (act_std + 1e-8)
                        # FIXME: and True for atari.
                        if self._predict_delta:
                            # Predict \Delta(s',s)
                            obs = batch['observations']
                            next_obs = batch['next_observations'] - batch['observations']
                        else:
                            if self.output_type == 'classification':
                                obs = batch['observations']
                                next_obs = batch['next_observations']
                            elif self.output_type == 'regression':
                                obs = (batch['observations'] - obs_mean) / (obs_std + 1e-8)
                                next_obs = (batch['next_observations'] - obs_mean) / (obs_std + 1e-8)

                        _inputs = np.hstack([obs, act])
                        _targets = next_obs
                        if self.predict_reward:
                            _targets = np.hstack((next_obs, batch['rewards'][:, None]))

                        acc_before += self.accuracy(_inputs, _targets)
                        train_loss += self.bnn.train_fn(
                            _inputs, _targets, kl_factor)
                        if i % int(np.ceil(itr_tot / 5.)) == 0:
                            self.plot_pred_imgs(_inputs, _targets, itr, i)
                        acc_after += self.accuracy(_inputs, _targets)

                    train_loss /= itr_tot
                    acc_before /= itr_tot
                    acc_after /= itr_tot

                    kl_factor *= self.replay_kl_schedule

            else:
                # NO REPLAY POOL
                # --------------
                # Here we should take the current batch of samples and shuffle
                # them for i.d.d. purposes.
                logger.log(
                    "Fitting dynamics model to current sample batch ...")
                lst_obs, lst_obs_nxt, lst_act, lst_rew = [], [], [], []
                for path in paths:
                    for i in xrange(len(path['observations']) - 1):
                        if i % self.kl_batch_size == 0:
                            obs, obs_nxt, act, rew = [], [], [], []
                            lst_obs.append(obs)
                            lst_obs_nxt.append(obs_nxt)
                            lst_act.append(act)
                            lst_rew.append(rew)
                        obs.append(path['observations'][i])
                        act.append(path['actions'][i])
                        rew.append(path['rewards_orig'][i])
                        if self._predict_delta:
                            # Predict \Delta(s',s)
                            obs_nxt.append(path['observations'][i + 1] - path['observations'][i])
                        else:
                            obs_nxt.append(path['observations'][i + 1])

                # Stack into input and target set.
                X_train = [np.hstack((obs, act)) for obs, act in zip(lst_obs, lst_act)]
                T_train = [np.hstack((obs_nxt, np.asarray(rew)[:, np.newaxis]))
                           for obs_nxt, rew in zip(lst_obs_nxt, lst_rew)]
                lst_surpr = [np.empty((_e.shape)) for _e in X_train]
                [_e.fill(np.nan) for _e in lst_surpr]

                acc_before = self.accuracy(np.vstack(X_train), np.vstack(T_train))

                # Do posterior chaining: this means that we update the model on each individual
                # minibatch and update the prior to the new posterior.
                count = 0
                lst_idx = np.arange(len(X_train))
                np.random.shuffle(lst_idx)
                loss_before, loss_after = 0., 0.
                for idx in lst_idx:
                    # Don't use kl_factor when using no replay pool. So here we form an outer
                    # loop around the individual minibatches, the model gets updated on each minibatch.
                    if itr > 0 and self.surprise_type == conv_bnn_vime.ConvBNNVIME.SurpriseType.COMPR and not self.second_order_update:
                        logp_before = self.bnn.fn_logp(X_train[idx], T_train[idx])
                    # Save old posterior as new prior.
                    self.bnn.save_params()
                    loss_before = float(self.bnn.fn_loss(X_train[idx], T_train[idx], 1.))
                    num_itr = int(np.ceil(float(self.num_sample_updates) / self.kl_batch_size))
                    for _ in xrange(num_itr):
                        train_loss = self.bnn.train_fn(X_train[idx], T_train[idx], 1.)
                        if np.isinf(train_loss) or np.isnan(train_loss):
                            import ipdb
                            ipdb.set_trace()
                        assert not np.isnan(train_loss)
                        assert not np.isinf(train_loss)
                        if count % int(np.ceil(len(X_train) * num_itr / float(self.kl_batch_size) / 5.)) == 0:
                            self.plot_pred_imgs(X_train[idx], T_train[idx], itr, count)
                        count += 1
                    loss_after = float(self.bnn.fn_loss(X_train[idx], T_train[idx], 1.))
                    if itr > 0 and self.surprise_type == conv_bnn_vime.ConvBNNVIME.SurpriseType.COMPR and not self.second_order_update:
                        # Samples will default path['KL'] to np.nan. It is filled in here.
                        logp_after = self.bnn.fn_logp(X_train[idx], T_train[idx])
                        lst_surpr[idx] = logp_after - logp_before
#                         if (lst_surpr[idx] < 0).any():
#                             print('\t\t >> ', float(loss_before), float(loss_after), float(lst_surpr[idx]))
#                         else:
#                             print(float(loss_before), float(loss_after), float(lst_surpr[idx]))

                if itr > 0 and self.surprise_type == conv_bnn_vime.ConvBNNVIME.SurpriseType.COMPR and not self.second_order_update:
                    # Make sure surprise >= 0
                    lst_surpr = np.concatenate(lst_surpr)
                    lst_surpr[lst_surpr < 0] = 0.
                    pc = 0
                    for path in paths:
                        _l = len(path['KL']) - 1
                        path['KL'] = np.append(lst_surpr[pc:pc + _l], lst_surpr[pc + _l - 1])
                        assert not np.isnan(path['KL']).any()
                        pc += _l

                acc_after = self.accuracy(np.vstack(X_train), np.vstack(T_train))

            # At this point, the dynamics model has been updated
            # according to new data, either from the replay pool
            # or from the current batch, using coarse- or fine-grained
            # posterior chaining.
            logger.log('Dynamics model updated.')

            logger.record_tabular('SurprFactor', kl_factor)
            logger.record_tabular('DynModel_SqErrBefore', acc_before)
            logger.record_tabular('DynModel_SqErrAfter', acc_after)
            logger.record_tabular('DynModel_TrainLoss', train_loss)

            samples_data = self.process_samples(itr, paths)

            self.env.log_diagnostics(paths)
            self.policy.log_diagnostics(paths)
            self.baseline.log_diagnostics(paths)
            self.optimize_policy(itr, samples_data)
            logger.log("Saving snapshot ...")
            params = self.get_itr_snapshot(itr, samples_data)
            paths = samples_data["paths"]
            if self.store_paths:
                params["paths"] = paths
            episode_rewards.extend(sum(p["rewards"]) for p in paths)
            episode_lengths.extend(len(p["rewards"]) for p in paths)
            params["episode_rewards"] = np.array(episode_rewards)
            params["episode_lengths"] = np.array(episode_lengths)
            params["algo"] = self
#             logger.save_itr_params(itr, params)
            logger.log("Saved.")
            logger.dump_tabular(with_prefix=False)
            logger.pop_prefix()
            if self.plot:
                self.update_plot()
                if self.pause_for_plot:
                    raw_input("Plotting evaluation run: Press Enter to "
                              "continue...")

        # Training complete: terminate environment.
        self.shutdown_worker()
        self.env.terminate()
        self.policy.terminate()

    def init_opt(self):
        """
        Initialize the optimization procedure. If using theano / cgt, this may
        include declaring all the variables and compiling functions
        """
        raise NotImplementedError

    def get_itr_snapshot(self, itr, samples_data):
        """
        Returns all the data that should be saved in the snapshot for this
        iteration.
        """
        raise NotImplementedError

    def optimize_policy(self, itr, samples_data):
        raise NotImplementedError

    def update_plot(self):
        if self.plot:
            plotter.update_plot(self.policy, self.max_path_length)

    def obtain_samples(self, itr):
        cur_params = self.policy.get_param_values()
        cur_dynamics_params = self.bnn.get_param_values()

        reward_mean = None
        reward_std = None
        if self.normalize_reward:
            # Compute running mean/std.
            reward_mean = np.mean(np.asarray(self._reward_mean))
            reward_std = np.mean(np.asarray(self._reward_std))

        # Mean/std obs/act based on replay pool.
        if self._dyn_pool_args['enable']:
            obs_mean, obs_std, act_mean, act_std = self.pool.mean_obs_act()
        else:
            obs_mean, obs_std, act_mean, act_std = 0, 1, 0, 1

        paths = parallel_sampler.sample_paths(
            policy_params=cur_params,
            dynamics_params=cur_dynamics_params,
            max_samples=self.batch_size,
            max_path_length=self.max_path_length,
            itr=itr,
            normalize_reward=self.normalize_reward,
            reward_mean=reward_mean,
            reward_std=reward_std,
            kl_batch_size=self.kl_batch_size,
            n_itr_update=self.n_itr_update,
            use_replay_pool=self._dyn_pool_args['enable'],
            obs_mean=obs_mean,
            obs_std=obs_std,
            act_mean=act_mean,
            act_std=act_std,
            second_order_update=self.second_order_update,
            predict_reward=self.predict_reward,
            surprise_type=self.surprise_type
        )

        if self.whole_paths:
            return paths
        else:
            paths_truncated = parallel_sampler.truncate_paths(
                paths, self.batch_size)
            return paths_truncated

    def process_samples(self, itr, paths):

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
                paths[i]['rewards'] = (paths[i]['rewards'] - reward_mean) / (reward_std + 1e-8)

        if itr > 0:
            kls = []
            for i in xrange(len(paths)):
                # We divide the KL by the number of weights in the network, to
                # get a more normalized surprise measure accross models.
                if self.surprise_type is not conv_bnn_vime.ConvBNNVIME.SurpriseType.COMPR:
                    # Don't normalize by weight division in case of compr gain.
                    kls.append(paths[i]['KL'])
                else:
                    kls.append(paths[i]['KL'] / float(self.num_weights))

            kls_flat = np.hstack(kls)

            logger.record_tabular('VIME_MeanSurpr', np.mean(kls_flat))
            logger.record_tabular('VIME_StdSurpr', np.std(kls_flat))
            logger.record_tabular('VIME_MinSurpr', np.min(kls_flat))
            logger.record_tabular('VIME_MaxSurpr', np.max(kls_flat))
            logger.record_tabular('VIME_MedianSurpr', np.median(kls_flat))
            logger.record_tabular('VIME_25percSurpr', np.percentile(kls_flat, 25))
            logger.record_tabular('VIME_75percSurpr', np.percentile(kls_flat, 75))
            logger.record_tabular('VIME_90percSurpr', np.percentile(kls_flat, 90))

            # Transform intrinsic rewards.
            if self.surprise_transform == BatchPolopt.SurpriseTransform.LOG:
                # Transform surprise into (positive) log space.
                for i in xrange(len(paths)):
                    kls[i] = np.log(1 + kls[i])
            elif self.surprise_transform == BatchPolopt.SurpriseTransform.CAP90PERC:
                perc90 = np.percentile(np.hstack(kls), 90)
                # Cap max KL for stabilization.
                for i in xrange(len(paths)):
                    kls[i] = np.minimum(kls[i], perc90)
            elif self.surprise_transform == BatchPolopt.SurpriseTransform.CAP1000:
                # Cap max KL for stabilization.
                for i in xrange(len(paths)):
                    kls[i] = np.minimum(kls[i], 1000)
            elif self.surprise_transform == BatchPolopt.SurpriseTransform.ZERO100:
                cap = np.percentile(kls_flat, 95)
                kls = [np.minimum(kl, cap) for kl in kls]
                kls_flat, lens = ungroup(kls)
                zerohunderd = stats.rankdata(kls_flat, "average") / len(kls_flat)
                kls = group(zerohunderd, lens)

            kls_flat = np.hstack(kls)

            if self.surprise_transform is not None:
                logger.record_tabular('VIME_MeanSurpr_transf', np.mean(kls_flat))
                logger.record_tabular('VIME_StdSurpr_transf', np.std(kls_flat))
                logger.record_tabular('VIME_MinSurpr_transf', np.min(kls_flat))
                logger.record_tabular('VIME_MaxSurpr_transf', np.max(kls_flat))
                logger.record_tabular('VIME_MedianSurpr_transf', np.median(kls_flat))
                logger.record_tabular('VIME_25percSurpr_transf', np.percentile(kls_flat, 25))
                logger.record_tabular('VIME_75percSurpr_transf', np.percentile(kls_flat, 75))
                logger.record_tabular('VIME_90percSurpr_transf', np.percentile(kls_flat, 90))

            # Normalize intrinsic rewards.
            if self.use_kl_ratio:
                if self.use_kl_ratio_q:
                    # Update kl Q
                    self.kl_previous.append(np.median(np.hstack(kls)))
                    previous_mean_kl = np.mean(np.asarray(self.kl_previous))
                    for i in xrange(len(kls)):
                        kls[i] = kls[i] / previous_mean_kl
                else:
                    median_KL_current_batch = np.median(np.hstack(kls))
                    for i in xrange(len(kls)):
                        kls[i] = kls[i] / median_KL_current_batch
                        # FIXME: inserted clip for stabilization.
                        kls[i] = np.minimum(kls[i], 100)

            kls_flat = np.hstack(kls)

            if self.use_kl_ratio:
                logger.record_tabular('VIME_MeanSurpr_norm', np.mean(kls_flat))
                logger.record_tabular('VIME_StdSurpr_norm', np.std(kls_flat))
                logger.record_tabular('VIME_MinSurpr_norm', np.min(kls_flat))
                logger.record_tabular('VIME_MaxSurpr_norm', np.max(kls_flat))
                logger.record_tabular('VIME_MedianSurpr_norm', np.median(kls_flat))
                logger.record_tabular('VIME_25percSurpr_norm', np.percentile(kls_flat, 25))
                logger.record_tabular('VIME_75percSurpr_norm', np.percentile(kls_flat, 75))
                logger.record_tabular('VIME_90percSurpr_norm', np.percentile(kls_flat, 90))

            # Add Surpr as intrinsic reward to external reward
            for i in xrange(len(paths)):
                paths[i]['rewards'] = paths[i]['rewards'] + self.eta * kls[i]

        else:
            logger.record_tabular('VIME_MeanSurpr', 0.)
            logger.record_tabular('VIME_StdSurpr', 0.)
            logger.record_tabular('VIME_MinSurpr', 0.)
            logger.record_tabular('VIME_MaxSurpr', 0.)
            logger.record_tabular('VIME_MedianSurpr', 0.)
            logger.record_tabular('VIME_25percSurpr', 0.)
            logger.record_tabular('VIME_75percSurpr', 0.)
            logger.record_tabular('VIME_90percSurpr', 0.)

            if self.surprise_transform is not None:
                logger.record_tabular('VIME_MeanSurpr_transf', 0.)
                logger.record_tabular('VIME_StdSurpr_transf', 0.)
                logger.record_tabular('VIME_MinSurpr_transf', 0.)
                logger.record_tabular('VIME_MaxSurpr_transf', 0.)
                logger.record_tabular('VIME_MedianSurpr_transf', 0.)
                logger.record_tabular('VIME_25percSurpr_transf', 0.)
                logger.record_tabular('VIME_75percSurpr_transf', 0.)
                logger.record_tabular('VIME_90percSurpr_transf', 0.)

            if self.use_kl_ratio:
                logger.record_tabular('VIME_MeanSurpr_norm', 0.)
                logger.record_tabular('VIME_StdSurpr_norm', 0.)
                logger.record_tabular('VIME_MinSurpr_norm', 0.)
                logger.record_tabular('VIME_MaxSurpr_norm', 0.)
                logger.record_tabular('VIME_MedianSurpr_norm', 0.)
                logger.record_tabular('VIME_25percSurpr_norm', 0.)
                logger.record_tabular('VIME_75percSurpr_norm', 0.)
                logger.record_tabular('VIME_90percSurpr_norm', 0.)

        baselines = []
        returns = []
        for path in paths:
            path_baselines = np.append(self.baseline.predict(path), 0)
            deltas = path["rewards"] + self.discount * path_baselines[1:] - path_baselines[:-1]
            path["advantages"] = special.discount_cumsum(deltas, self.discount * self.gae_lambda)
            # FIXME: does this have to be rewards_orig or rewards? DEFAULT:
            # rewards_orig
            path["returns"] = special.discount_cumsum(path["rewards_orig"], self.discount)
            baselines.append(path_baselines[:-1])
            returns.append(path["returns"])

        if not self.policy.recurrent:
            observations = tensor_utils.concat_tensor_list(
                [path["observations"] for path in paths])
            actions = tensor_utils.concat_tensor_list(
                [path["actions"] for path in paths])
            rewards = tensor_utils.concat_tensor_list(
                [path["rewards"] for path in paths])
            advantages = tensor_utils.concat_tensor_list(
                [path["advantages"] for path in paths])
            env_infos = tensor_utils.concat_tensor_dict_list(
                [path["env_infos"] for path in paths])
            agent_infos = tensor_utils.concat_tensor_dict_list(
                [path["agent_infos"] for path in paths])

            if self.center_adv:
                advantages = util.center_advantages(advantages)

            if self.positive_adv:
                advantages = util.shift_advantages_to_positive(advantages)

            average_discounted_return = np.mean([path["returns"][0] for path in paths])

            undiscounted_returns = [sum(path["rewards_orig"]) for path in paths]

            ent = np.mean(self.policy.distribution.entropy(agent_infos))

            ev = special.explained_variance_1d(
                np.concatenate(baselines),
                np.concatenate(returns)
            )

            samples_data = dict(
                observations=observations,
                actions=actions,
                rewards=rewards,
                advantages=advantages,
                env_infos=env_infos,
                agent_infos=agent_infos,
                paths=paths,
            )
        else:
            max_path_length = max([len(path["advantages"]) for path in paths])

            # make all paths the same length (pad extra advantages with 0)
            obs = [path["observations"] for path in paths]
            obs = np.array(
                [tensor_utils.pad_tensor(ob, max_path_length) for ob in obs])

            if self.center_adv:
                raw_adv = np.concatenate(
                    [path["advantages"] for path in paths])
                adv_mean = np.mean(raw_adv)
                adv_std = np.std(raw_adv) + 1e-8
                adv = [
                    (path["advantages"] - adv_mean) / adv_std for path in paths]
            else:
                adv = [path["advantages"] for path in paths]

            adv = np.array(
                [tensor_utils.pad_tensor(a, max_path_length) for a in adv])

            actions = [path["actions"] for path in paths]
            actions = np.array(
                [tensor_utils.pad_tensor(a, max_path_length) for a in actions])

            rewards = [path["rewards"] for path in paths]
            rewards = np.array(
                [tensor_utils.pad_tensor(r, max_path_length) for r in rewards])

            agent_infos = [path["agent_infos"] for path in paths]
            agent_infos = tensor_utils.stack_tensor_dict_list(
                [tensor_utils.pad_tensor_dict(
                    p, max_path_length) for p in agent_infos]
            )

            env_infos = [path["env_infos"] for path in paths]
            env_infos = tensor_utils.stack_tensor_dict_list(
                [tensor_utils.pad_tensor_dict(
                    p, max_path_length) for p in env_infos]
            )

            valids = [np.ones_like(path["returns"]) for path in paths]
            valids = np.array(
                [tensor_utils.pad_tensor(v, max_path_length) for v in valids])

            average_discounted_return = \
                np.mean([path["returns"][0] for path in paths])

            undiscounted_returns = [
                sum(path["rewards_orig"]) for path in paths]

            ent = np.mean(self.policy.distribution.entropy(agent_infos))

            ev = special.explained_variance_1d(
                np.concatenate(baselines),
                np.concatenate(returns)
            )

            samples_data = dict(
                observations=obs,
                actions=actions,
                advantages=adv,
                rewards=rewards,
                valids=valids,
                agent_infos=agent_infos,
                env_infos=env_infos,
                paths=paths,
            )

        logger.log("fitting baseline...")
        self.baseline.fit(paths)
        logger.log("fitted")

        average_reward = np.mean(
            [np.mean(path["rewards_orig"]) for path in paths])
        min_reward = np.min(
            [np.min(path["rewards_orig"]) for path in paths])
        max_reward = np.max(
            [np.min(path["rewards_orig"]) for path in paths])

        logger.record_tabular('Iteration', itr)
        logger.record_tabular('AverageReward', average_reward)
        logger.record_tabular('MinReward', min_reward)
        logger.record_tabular('MaxReward', max_reward)
        logger.record_tabular('AverageDiscountedReturn',
                              average_discounted_return)
        logger.record_tabular('AverageReturn', np.mean(undiscounted_returns))
        logger.record_tabular('ExplainedVariance', ev)
        logger.record_tabular('NumTrajs', len(paths))
        logger.record_tabular('Entropy', ent)
        logger.record_tabular('Perplexity', np.exp(ent))
        logger.record_tabular('StdReturn', np.std(undiscounted_returns))
        logger.record_tabular('MaxReturn', np.max(undiscounted_returns))
        logger.record_tabular('MinReturn', np.min(undiscounted_returns))
        logger.record_tabular('VIME_eta', self.eta)
        if self.update_likelihood_sd and self.output_type == 'regression':
            logger.record_tabular(
                'LikelihoodStd', self.bnn.likelihood_sd.eval())

        return samples_data
