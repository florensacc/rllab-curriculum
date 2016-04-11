import numpy as np
from rllab.algos.base import RLAlgorithm
from rllab.sampler import parallel_sampler
from rllab.misc import special
from rllab.misc import tensor_utils
from rllab.algos import util
import rllab.misc.logger as logger
import rllab.plotter as plotter

# exploration imports
# -------------------
import theano
import lasagne
from collections import deque
import time
from sandbox.rein.dynamics_models.nn_uncertainty import prob_nn
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

    @property
    def size(self):
        return self._size


class BatchPolopt(RLAlgorithm):
    """
    Base class for batch sampling-based policy optimization methods.
    This includes various policy gradient methods like vpg, npg, ppo, trpo, etc.
    """

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
            prior_sd=0.05,
            use_kl_ratio=False,
            kl_q_len=5,
            reverse_update_kl=False,
            symbolic_prior_kl=True,  # leave as is
            use_reverse_kl_reg=False,
            reverse_kl_reg_factor=0.2,
            use_replay_pool=True,
            replay_pool_size=100000,
            min_pool_size=500,
            n_updates_per_sample=500,
            pool_batch_size=10,
            eta_discount=1.0,
            n_itr_update=5,
            whole_paths=False,
            **kwargs
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
        self.whole_paths = whole_paths


        # Set exploration params
        # ----------------------
        self.eta = eta
        self.snn_n_samples = snn_n_samples
        self.prior_sd = prior_sd
        self.use_kl_ratio = use_kl_ratio
        self.kl_q_len = kl_q_len
        self.reverse_update_kl = reverse_update_kl
        self.symbolic_prior_kl = symbolic_prior_kl
        self.use_reverse_kl_reg = use_reverse_kl_reg
        self.reverse_kl_reg_factor = reverse_kl_reg_factor
        self.use_replay_pool = use_replay_pool
        self.replay_pool_size = replay_pool_size
        self.min_pool_size = min_pool_size
        self.n_updates_per_sample = n_updates_per_sample
        self.pool_batch_size = pool_batch_size
        self.eta_discount = eta_discount
        self.n_itr_update = n_itr_update
        # ----------------------

    def start_worker(self):
        parallel_sampler.populate_task(self.env, self.policy)
        if self.plot:
            plotter.init_plot(self.env, self.policy)

    def shutdown_worker(self):
        pass

    def train(self):

        # Uncertainty neural network (UNN) initialization.
        # ------------------------------------------------
        batch_size = 1
        n_batches = 5  # temp

        obs_dim = np.sum(self.env.observation_space.shape)
        act_dim = 1  # mdp.action_dim
        # double pendulum: in=7 out=6
        self.pnn = prob_nn.ProbNN(
            n_in=(obs_dim + act_dim),
            n_hidden=[32],
            n_out=obs_dim,
            n_batches=n_batches,
            layers_type=[1, 1],
            trans_func=lasagne.nonlinearities.rectify,
            out_func=lasagne.nonlinearities.linear,
            batch_size=batch_size,
            n_samples=self.snn_n_samples,
            type='regression',
            prior_sd=self.prior_sd,
            reverse_update_kl=self.reverse_update_kl,
            symbolic_prior_kl=self.symbolic_prior_kl,
            use_reverse_kl_reg=self.use_reverse_kl_reg,
            reverse_kl_reg_factor=self.reverse_kl_reg_factor
        )

        if self.use_kl_ratio:
            # Add Queue here to keep track of N last kl values, compute average
            # over them and divide current kl values by it. This counters the
            # exploding kl value problem.
            self.kl_previous = deque(maxlen=self.kl_q_len)

        print("Building UNN model (eta={}) ...".format(self.eta))
        start_time = time.time()
        # Build symbolic network architecture.
        self.pnn.build_network()
        # Build all symbolic stuff around architecture, e.g., loss, prediction
        # functions, training functions,...
        self.pnn.build_model()
        print("Model built ({:.1f} sec).".format((time.time() - start_time)))

        if self.use_replay_pool:
            pool = SimpleReplayPool(
                max_pool_size=self.replay_pool_size,
                observation_shape=self.env.observation_space.shape,
                action_dim=1,  # mdp.action_dim,
            )
        # ------------------------------------------------

        self.start_worker()
        self.init_opt()
        episode_rewards = []
        episode_lengths = []
        for itr in xrange(self.start_itr, self.n_itr):
            logger.push_prefix('itr #%d | ' % itr)
            paths = self.obtain_samples(itr)
            samples_data = self.process_samples(itr, paths)

            # Exploration code
            # ----------------
            if self.use_replay_pool:
                # Fill replay pool.
                logger.log("Fitting dynamics model using replay pool ...")
                for path in samples_data['paths']:
                    path_len = len(path['rewards'])
                    for i in xrange(path_len):
                        obs = path['observations'][i]
                        act = path['actions'][i]
                        rew = path['rewards'][i]
                        term = (i == path_len - 1)
                        pool.add_sample(obs, act, rew, term)

                # Now we train the dynamics model using the replay pool; only
                # if pool is large enough.
                if pool.size >= self.min_pool_size:
                    _inputss = []
                    _targetss = []
                    for _ in xrange(self.n_updates_per_sample):
                        batch = pool.random_batch(self.pool_batch_size)
                        _inputs = np.hstack(
                            [batch['observations'], batch['actions']])
                        _targets = batch['next_observations']
                        _inputss.append(_inputs)
                        _targetss.append(_targets)

                    _out = self.pnn.pred_fn(np.vstack(_inputss))
                    old_acc = np.square(_out - np.vstack(_targetss))
                    old_acc = np.mean(old_acc)

                    for _inputs, _targets in zip(_inputss, _targetss):
                        self.pnn.train_fn(_inputs, _targets)

                    _out = self.pnn.pred_fn(_inputs)

                    _out = self.pnn.pred_fn(np.vstack(_inputss))
                    new_acc = np.square(_out - np.vstack(_targetss))
                    new_acc = np.mean(new_acc)
            # ----------------

            self.env.log_diagnostics(paths)
            self.policy.log_diagnostics(paths)
            self.baseline.log_diagnostics(paths)
            self.optimize_policy(itr, samples_data)
            logger.log("saving snapshot...")
            params = self.get_itr_snapshot(itr, samples_data)
            paths = samples_data["paths"]
            if self.store_paths:
                params["paths"] = paths
            episode_rewards.extend(sum(p["rewards"]) for p in paths)
            episode_lengths.extend(len(p["rewards"]) for p in paths)
            params["episode_rewards"] = np.array(episode_rewards)
            params["episode_lengths"] = np.array(episode_lengths)
            params["algo"] = self
            logger.save_itr_params(itr, params)
            logger.log("saved")
            logger.dump_tabular(with_prefix=False)
            logger.pop_prefix()
            if self.plot:
                self.update_plot()
                if self.pause_for_plot:
                    raw_input("Plotting evaluation run: Press Enter to "
                              "continue...")
        self.shutdown_worker()

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
        paths = parallel_sampler.sample_paths(
            policy_params=cur_params,
            max_samples=self.batch_size,
            max_path_length=self.max_path_length,
        )
        if self.whole_paths:
            return paths
        else:
            paths_truncated = parallel_sampler.truncate_paths(paths, self.batch_size)
            return paths_truncated

    def process_samples(self, itr, paths):

        # Computing intrinsic rewards.
        # ----------------------------
        logger.log("Computing intrinsic rewards ...")
        # Compute sample Bellman error.
        obs = np.vstack([path['observations'] for path in paths])
        act = np.vstack([path['actions'] for path in paths])
        rew = np.hstack([path['rewards'] for path in paths])

        # inputs = (o,a), target = o'
        obs_nxt = np.vstack([obs[1:], obs[-1]])
        _inputs = np.hstack([obs, act])
        _targets = obs_nxt

        _out = self.pnn.pred_fn(_inputs)

        old_acc = np.square(_out - _targets)
        old_acc = np.mean(old_acc)

        kl = np.zeros(rew.shape)

        for i in xrange(obs.shape[0]):
            # Save old params for every update.
            self.pnn.save_old_params()

            # Update model weights based on current minibatch.
            for _ in xrange(self.n_itr_update):
                self.pnn.train_update_fn(
                    _inputs[i][None, :], _targets[i][None, :])

            # Calculate current minibatch KL.
            kl_div = float(self.pnn.f_kl_div_closed_form())
            kl[i] = kl_div

            # If using replay pool, undo updates.
            if self.use_replay_pool:
                self.pnn.reset_to_old_params()

        if self.use_kl_ratio:
            logger.log(str(self.kl_previous))
            self.kl_previous.append(np.mean(kl))
            kl = kl / np.mean(np.asarray(self.kl_previous))

        _out = self.pnn.pred_fn(_inputs)

        new_acc = np.square(_out - _targets)
        new_acc = np.mean(new_acc)

        self.eta *= self.eta_discount
        kl = self.eta * np.hstack([0., kl])
        n_samples = 0
        for i in xrange(len(paths)):
            path_length = len(paths[i]['rewards'])
            paths[i]['rewards_orig'] = np.array(paths[i]['rewards'])
            paths[i]['rewards'] = paths[i]['rewards'] + \
                kl[n_samples:n_samples + path_length]
            n_samples += path_length

        logger.log("fitted")
        # ----------------------------

        baselines = []
        returns = []
        for path in paths:
            path_baselines = np.append(self.baseline.predict(path), 0)
            deltas = path["rewards"] + \
                self.discount * path_baselines[1:] - \
                path_baselines[:-1]
            path["advantages"] = special.discount_cumsum(
                deltas, self.discount * self.gae_lambda)
            path["returns"] = special.discount_cumsum(
                path["rewards_orig"], self.discount)
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

            undiscounted_returns = [sum(path["rewards"]) for path in paths]

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

        logger.record_tabular('Iteration', itr)
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
        logger.record_tabular('SNN_MeanKL', np.mean(kl))
        logger.record_tabular('SNN_StdKL', np.std(kl))
        logger.record_tabular('SNN_MinKL', np.min(kl))
        logger.record_tabular('SNN_MaxKL', np.max(kl))
        logger.record_tabular('SNN_DynModelSqLossBefore', old_acc)
        logger.record_tabular('SNN_DynModelSqLossAfter', new_acc)
        logger.record_tabular('eta', self.eta)

        return samples_data
