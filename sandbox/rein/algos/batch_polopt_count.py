import numpy as np
from rllab.algos.base import RLAlgorithm
from sandbox.rein.sampler import parallel_sampler_count as parallel_sampler
from rllab.misc import special
from rllab.misc import tensor_utils
from rllab.algos import util
import rllab.misc.logger as logger
import rllab.plotter as plotter
from sandbox.rein.dynamics_models.utils import iterate_minibatches, group, ungroup
from scipy import stats, misc
from sandbox.rein.dynamics_models.utils import enum
from sandbox.rein.algos.replay_pool import ReplayPool
from sandbox.rein.dynamics_models.bnn import conv_bnn_vime

# Nonscientific printing of numpy arrays.
np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)


class BatchPolopt(RLAlgorithm):
    """
    Base class for batch sampling-based policy optimization methods.
    This includes various policy gradient methods like vpg, npg, ppo, trpo, etc.
    This one is specifically created for using discrete embedding-based counts.
    """

    # Enums
    SurpriseTransform = enum(
        CAP1000='cap at 1000',
        LOG='log(1+surprise)',
        ZERO100='0-100',
        CAP90PERC='cap at 90th percentile',
        CAP99PERC='cap at 99th percentile')

    def __init__(
            self,
            env,
            policy,
            baseline,
            autoenc,
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
            store_paths=False,
            # exploration params
            eta=1.,
            use_kl_ratio=False,
            num_sample_updates=1,
            n_itr_update=5,
            reward_alpha=0.001,
            kl_alpha=0.001,
            kl_batch_size=1,
            surprise_transform=None,
            replay_kl_schedule=1.0,
            predict_delta=False,
            dyn_pool_args=None,
            num_seq_frames=1,
            **kwargs
    ):
        """
        :param env: Environment
        :param policy: Policy
        :param baseline: Baseline
        :param dyn_mdl: dynamics model
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

        # Set models
        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.autoenc = autoenc

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

        if dyn_pool_args is None:
            dyn_pool_args = dict(size=100000, min_size=10, batch_size=32)

        self.eta = eta
        self.use_kl_ratio = use_kl_ratio
        self.num_sample_updates = num_sample_updates
        self.n_itr_update = n_itr_update
        self.reward_alpha = reward_alpha
        self.kl_alpha = kl_alpha
        self.kl_batch_size = kl_batch_size
        self.surprise_transform = surprise_transform
        self.replay_kl_schedule = replay_kl_schedule
        self._predict_delta = predict_delta
        self._dyn_pool_args = dyn_pool_args
        self._num_seq_frames = num_seq_frames

        observation_dtype = "uint8"
        self.pool = ReplayPool(
            max_pool_size=self._dyn_pool_args['size'],
            # self.env.observation_space.shape,
            observation_shape=(self.env.observation_space.flat_dim,),
            action_dim=self.env.action_dim,
            observation_dtype=observation_dtype,
            num_seq_frames=self._num_seq_frames,
            **self._dyn_pool_args
        )

    def start_worker(self):
        parallel_sampler.populate_task(self.env, self.policy, self.autoenc)
        if self.plot:
            plotter.init_plot(self.env, self.policy)

    def shutdown_worker(self):
        pass

    def accuracy(self, _inputs, _targets):
        acc = 0.
        for batch in iterate_minibatches(_inputs, _targets, 1000, shuffle=False):
            _i, _t, _ = batch
            _o = self.autoenc.pred_fn(_i)
            if self.autoenc.output_type == conv_bnn_vime.ConvBNNVIME.OutputType.CLASSIFICATION:
                _o_s = _o[:, :-1]
                _o_r = _o[:, -1]
                _o_s = _o_s.reshape((-1, np.prod(self.autoenc.state_dim), self.autoenc.num_classes))
                _o_s = np.argmax(_o_s, axis=2)
                acc += np.sum(np.abs(_o_s - _t[:, :-1])) + np.sum(np.abs(_o_r - _t[:, -1]))
            else:
                acc += np.sum(np.square(_o - _t))
        return acc / _inputs.shape[0]

    def plot_pred_imgs(self, inputs, targets, itr, count):
        # try:
        # This is specific to Atari.
        import matplotlib.pyplot as plt
        if not hasattr(self, '_fig'):
            self._fig = plt.figure()
            self._fig_1 = self._fig.add_subplot(241)
            plt.tick_params(axis='both', which='both', bottom='off', top='off',
                            labelbottom='off', right='off', left='off', labelleft='off')
            self._fig_2 = self._fig.add_subplot(242)
            plt.tick_params(axis='both', which='both', bottom='off', top='off',
                            labelbottom='off', right='off', left='off', labelleft='off')
            self._fig_3 = self._fig.add_subplot(243)
            plt.tick_params(axis='both', which='both', bottom='off', top='off',
                            labelbottom='off', right='off', left='off', labelleft='off')
            self._fig_4 = self._fig.add_subplot(244)
            plt.tick_params(axis='both', which='both', bottom='off', top='off',
                            labelbottom='off', right='off', left='off', labelleft='off')
            self._fig_5 = self._fig.add_subplot(245)
            plt.tick_params(axis='both', which='both', bottom='off', top='off',
                            labelbottom='off', right='off', left='off', labelleft='off')
            self._fig_6 = self._fig.add_subplot(246)
            plt.tick_params(axis='both', which='both', bottom='off', top='off',
                            labelbottom='off', right='off', left='off', labelleft='off')
            self._fig_7 = self._fig.add_subplot(247)
            plt.tick_params(axis='both', which='both', bottom='off', top='off',
                            labelbottom='off', right='off', left='off', labelleft='off')
            self._fig_8 = self._fig.add_subplot(248)
            plt.tick_params(axis='both', which='both', bottom='off', top='off',
                            labelbottom='off', right='off', left='off', labelleft='off')
            self._im1, self._im2, self._im3, self._im4 = None, None, None, None
            self._im5, self._im6, self._im7, self._im8 = None, None, None, None

        idx = np.random.randint(0, inputs.shape[0], 1)
        sanity_pred = self.autoenc.pred_fn(inputs)
        input_im = inputs[:, :-self.env.spec.action_space.flat_dim]
        lst_input_im = [
            input_im[idx, i * np.prod(self.autoenc.state_dim):(i + 1) * np.prod(self.autoenc.state_dim)].reshape(
                self.autoenc.state_dim).transpose(1, 2, 0)[:, :, 0] * 256. for i in
            range(self._num_seq_frames)]
        input_im = input_im[:, -np.prod(self.autoenc.state_dim):]
        input_im = input_im[idx, :].reshape(self.autoenc.state_dim).transpose(1, 2, 0)[:, :, 0]
        sanity_pred_im = sanity_pred[idx, :-1]
        if self.autoenc.output_type == self.autoenc.OutputType.CLASSIFICATION:
            sanity_pred_im = sanity_pred_im.reshape((-1, self.autoenc.num_classes))
            sanity_pred_im = np.argmax(sanity_pred_im, axis=1)
        sanity_pred_im = sanity_pred_im.reshape(self.autoenc.state_dim).transpose(1, 2, 0)[:, :, 0]
        target_im = targets[idx, :-1].reshape(self.autoenc.state_dim).transpose(1, 2, 0)[:, :, 0]

        if self._predict_delta:
            sanity_pred_im += input_im
            target_im += input_im

        if self.autoenc.output_type == self.autoenc.OutputType.CLASSIFICATION:
            sanity_pred_im = sanity_pred_im.astype(float) / float(self.autoenc.num_classes)
            target_im = target_im.astype(float) / float(self.autoenc.num_classes)
            input_im = input_im.astype(float) / float(self.autoenc.num_classes)
            for i in range(len(lst_input_im)):
                lst_input_im[i] = lst_input_im[i].astype(float) / float(self.autoenc.num_classes)

        sanity_pred_im *= 256.
        sanity_pred_im = np.around(sanity_pred_im).astype(int)
        target_im *= 256.
        target_im = np.around(target_im).astype(int)
        err = (256 - np.abs(target_im - sanity_pred_im) * 100.)
        input_im *= 256.
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
            self._im5 = self._fig_5.imshow(
                lst_input_im[0], interpolation='none', cmap='Greys_r', vmin=0, vmax=255)
            self._im6 = self._fig_6.imshow(
                lst_input_im[1], interpolation='none', cmap='Greys_r', vmin=0, vmax=255)
            self._im7 = self._fig_7.imshow(
                lst_input_im[2], interpolation='none', cmap='Greys_r', vmin=0, vmax=255)
            self._im8 = self._fig_8.imshow(
                lst_input_im[3], interpolation='none', cmap='Greys_r', vmin=0, vmax=255)

        else:
            self._im1.set_data(input_im)
            self._im2.set_data(target_im)
            self._im3.set_data(sanity_pred_im)
            self._im4.set_data(err)
            self._im5.set_data(lst_input_im[0])
            self._im6.set_data(lst_input_im[1])
            self._im7.set_data(lst_input_im[2])
            self._im8.set_data(lst_input_im[3])
        plt.savefig(
            logger._snapshot_dir + '/dynpred_img_{}_{}.png'.format(itr, count), bbox_inches='tight')
        # except Exception:
        #     pass

    def train(self):

        self.start_worker()
        self.init_opt()

        episode_rewards, episode_lengths = [], []
        acc_before, acc_after, train_loss = 0., 0., 0.

        # KL rescaling factor for replay pool-based training.
        kl_factor = 1.0

        # ATTENTION: important to know when 'rewards' and 'rewards_orig' needs
        # to be used!
        for itr in range(self.start_itr, self.n_itr):
            logger.push_prefix('itr #%d | ' % itr)

            # Sample trajectories.
            paths = self.obtain_samples()

            # Fill replay pool with samples of current batch. Exclude the
            # last one.
            logger.log("Fitting dynamics model using replay pool ...")
            for path in paths:
                path_len = len(path['rewards'])
                for i in range(path_len):
                    obs = (path['observations'][i] * self.autoenc.num_classes).astype(int)
                    act = path['actions'][i]
                    rew_orig = path['rewards_orig'][i]
                    term = (i == path_len - 1)
                    self.pool.add_sample(obs, act, rew_orig, term)

            # Now we train the dynamics model using the replay self.pool; only
            # if self.pool is large enough.
            if self.pool.size >= self._dyn_pool_args['min_size']:
                acc_before, acc_after, train_loss = 0., 0., 0.
                itr_tot = int(
                    np.ceil(self.num_sample_updates * float(self.batch_size) / self._dyn_pool_args['batch_size']))

                for _ in range(20):
                    batch = self.pool.random_batch(self._dyn_pool_args['batch_size'])
                    _x = np.hstack([batch['observations'], batch['actions']])
                    _y = np.hstack([batch['next_observations'], batch['rewards'][:, np.newaxis]])
                    acc_before += self.accuracy(_x, _y)
                acc_before /= 20.

                for i in range(itr_tot):

                    batch = self.pool.random_batch(self._dyn_pool_args['batch_size'])

                    _x = np.hstack([batch['observations'], batch['actions']])
                    if self._predict_delta:
                        _y = np.hstack(
                            [(batch['next_observations'] - batch['observations']), batch['rewards'][:, np.newaxis]])
                    else:
                        _y = np.hstack([batch['next_observations'], batch['rewards'][:, np.newaxis]])

                    _tl = self.autoenc.train_fn(_x, _y, 0 * kl_factor)
                    train_loss += _tl
                    if i % int(np.ceil(itr_tot / 3.)) == 0:
                        self.plot_pred_imgs(_x, _y, itr, i)

                for _ in range(20):
                    batch = self.pool.random_batch(self._dyn_pool_args['batch_size'])
                    _x = np.hstack([batch['observations'], batch['actions']])
                    _y = np.hstack([batch['next_observations'], batch['rewards'][:, np.newaxis]])
                    acc_after += self.accuracy(_x, _y)
                acc_after /= 20.

                train_loss /= itr_tot

                kl_factor *= self.replay_kl_schedule

            # At this point, the dynamics model has been updated
            # according to new data, either from the replay pool
            # or from the current batch, using coarse- or fine-grained
            # posterior chaining.
            logger.log('Dynamics model updated.')

            logger.record_tabular('SurprFactor', kl_factor)
            logger.record_tabular('DynModel_SqErrBefore', acc_before)
            logger.record_tabular('DynModel_SqErrAfter', acc_after)
            logger.record_tabular('DynModel_TrainLoss', train_loss)

            # Here we should extract discrete embedding from samples and use it for updating count table.


            # Postprocess trajectory data.
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
                    input("Plotting evaluation run: Press Enter to "
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

    def obtain_samples(self):
        cur_params = self.policy.get_param_values()

        paths = parallel_sampler.sample_paths(
            policy_params=cur_params,
            max_samples=self.batch_size,
            max_path_length=self.max_path_length,
            num_seq_frames=self._num_seq_frames
        )

        if self.whole_paths:
            return paths
        else:
            paths_truncated = parallel_sampler.truncate_paths(
                paths, self.batch_size)
            return paths_truncated

    def process_samples(self, itr, paths):

        kls = []
        for i in xrange(len(paths)):
            # We divide the KL by the number of weights in the network, to
            # get a more normalized surprise measure accross models.
            kls.append(paths[i]['KL'])

        kls_flat = np.hstack(kls)

        logger.record_tabular('S_avg', np.mean(kls_flat))
        logger.record_tabular('S_std', np.std(kls_flat))
        logger.record_tabular('S_min', np.min(kls_flat))
        logger.record_tabular('S_max', np.max(kls_flat))
        logger.record_tabular('S_25p', np.percentile(kls_flat, 25))
        logger.record_tabular('S_med', np.median(kls_flat))
        logger.record_tabular('S_75p', np.percentile(kls_flat, 75))
        logger.record_tabular('S_90p', np.percentile(kls_flat, 90))
        logger.record_tabular('S_99p', np.percentile(kls_flat, 99))

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
        elif self.surprise_transform == BatchPolopt.SurpriseTransform.CAP99PERC:
            perc99 = np.percentile(np.hstack(kls), 99)
            # Cap max KL for stabilization.
            for i in xrange(len(paths)):
                kls[i] = np.minimum(kls[i], perc99)
        elif self.surprise_transform == BatchPolopt.SurpriseTransform.CAP1000:
            # Cap max KL for stabilization.
            for i in xrange(len(paths)):
                kls[i] = np.minimum(kls[i], 1000)
        elif self.surprise_transform == BatchPolopt.SurpriseTransform.ZERO100:
            cap = np.percentile(kls_flat, 100)
            kls = [np.minimum(kl, cap) for kl in kls]
            kls_flat, lens = ungroup(kls)
            zerohunderd = stats.rankdata(kls_flat, "average") / len(kls_flat)
            kls = group(zerohunderd, lens)

        kls_flat = np.hstack(kls)

        # Add Surpr as intrinsic reward to external reward
        for i in xrange(len(paths)):
            paths[i]['rewards'] = paths[i]['rewards'] + self.eta * kls[i]

        baselines = []
        returns = []
        for path in paths:
            path_baselines = np.append(self.baseline.predict(path), 0)
            deltas = path["rewards"] + self.discount * path_baselines[1:] - path_baselines[:-1]
            path["advantages"] = special.discount_cumsum(deltas, self.discount * self.gae_lambda)
            # FIXME: does this have to be rewards_orig or rewards? DEFAULT:
            # rewards_orig
            # If we use rewards, rather than rewards_orig, we include the intrinsic reward in the baseline.
            path["returns"] = special.discount_cumsum(path["rewards"], self.discount)
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
            valids = np.array([tensor_utils.pad_tensor(v, max_path_length) for v in valids])

            average_discounted_return = np.mean([path["returns"][0] for path in paths])

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
        if self.autoenc.update_likelihood_sd and self.autoenc.output_type == 'regression':
            logger.record_tabular(
                'LikelihoodStd', self.autoenc.likelihood_sd.eval())

        return samples_data
