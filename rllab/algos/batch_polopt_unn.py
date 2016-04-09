import numpy as np
from rllab.algo.base import RLAlgorithm
from rllab.sampler import parallel_sampler
from rllab.misc import autoargs
from rllab.misc.ext import extract
from rllab.misc.special import explained_variance_1d, discount_cumsum
from rllab.algo.util import center_advantages, shift_advantages_to_positive
import rllab.misc.logger as logger
import rllab.plotter as plotter
from rllab.model.nn_uncertainty import prob_nn as prob_nn
import lasagne
import time
import theano
from collections import deque

G = parallel_sampler.G


def worker_inject_baseline(baseline):
    G.baseline = baseline


def worker_update_baseline(params):
    G.baseline.set_param_values(params, trainable=True)


def worker_retrieve_paths():
    return G.paths


def retrieve_paths():
    return sum(parallel_sampler.run_map(worker_retrieve_paths), [])


def worker_compute_paths_returns(opt):
    for path in G.paths:
        path["returns"] = discount_cumsum(path["rewards_orig"], opt.discount)


def worker_retrieve_samples_data():
    return G.samples_data


def aggregate_samples_data():
    samples_datas = parallel_sampler.run_map(worker_retrieve_samples_data)

    observations, states, pdists, actions, advantages, paths = extract(
        samples_datas,
        "observations", "states", "pdists", "actions", "advantages", "paths"
    )
    return dict(
        observations=np.concatenate(observations),
        states=np.concatenate(states),
        pdists=np.concatenate(pdists),
        actions=np.concatenate(actions),
        advantages=np.concatenate(advantages),
        paths=sum(paths, []),
    )


def worker_process_paths(opt):
    try:
        baselines = []
        returns = []
        for path in G.paths:
            path_baselines = np.append(G.baseline.predict(path), 0)
            deltas = path["rewards"] + \
                opt.discount * path_baselines[1:] - \
                path_baselines[:-1]
            path["advantages"] = discount_cumsum(
                deltas, opt.discount * opt.gae_lambda)
            baselines.append(path_baselines[:-1])
            returns.append(path["returns"])

        observations = np.vstack([path["observations"] for path in G.paths])
        states = np.vstack([path["states"] for path in G.paths])
        pdists = np.vstack([path["pdists"] for path in G.paths])
        actions = np.vstack([path["actions"] for path in G.paths])
        advantages = np.concatenate(
            [path["advantages"] for path in G.paths])

        if opt.center_adv:
            advantages = center_advantages(advantages)

        if opt.positive_adv:
            advantages = shift_advantages_to_positive(advantages)

        G.samples_data = dict(
            observations=observations,
            states=states,
            pdists=pdists,
            actions=actions,
            advantages=advantages,
            paths=G.paths,
        )

        average_discounted_return = \
            np.mean([path["returns"][0] for path in G.paths])

        undiscounted_returns = [sum(path["rewards_orig"]) for path in G.paths]

        return dict(
            average_discounted_return=average_discounted_return,
            average_return=np.mean(undiscounted_returns),
            std_return=np.std(undiscounted_returns),
            max_return=np.max(undiscounted_returns),
            min_return=np.min(undiscounted_returns),
            num_trajs=len(G.paths),
            ent=G.policy.compute_entropy(pdists),
            ev=explained_variance_1d(
                np.concatenate(baselines),
                np.concatenate(returns)
            )
        )
    except Exception as e:
        print e
        import traceback
        traceback.print_exc()
        raise


class SimpleReplayPool(object):

    def __init__(
            self, max_pool_size, observation_shape, action_dim,
            observation_dtype=theano.config.floatX,
            action_dtype=theano.config.floatX):
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

    @autoargs.arg("n_itr", type=int,
                  help="Number of iterations.")
    @autoargs.arg("start_itr", type=int,
                  help="Starting iteration.")
    @autoargs.arg("batch_size", type=int,
                  help="Number of samples per iteration.")
    @autoargs.arg("max_path_length", type=int,
                  help="Maximum length of a single rollout.")
    @autoargs.arg("whole_paths", type=bool,
                  help="Make sure that the samples contain whole "
                       "trajectories, even if the actual batch size is "
                       "slightly larger than the specified batch_size.")
    @autoargs.arg("discount", type=float,
                  help="Discount.")
    @autoargs.arg("gae_lambda", type=float,
                  help="Lambda used for generalized advantage estimation.")
    @autoargs.arg("center_adv", type=bool,
                  help="Whether to rescale the advantages so that they have "
                       "mean 0 and standard deviation 1")
    @autoargs.arg("positive_adv", type=bool,
                  help="Whether to shift the advantages so that they are "
                       "always positive. When used in conjunction with center adv"
                       "the advantages will be standardized before shifting")
    @autoargs.arg("record_states", type=bool,
                  help="Whether to record states when sampling")
    @autoargs.arg("store_paths", type=bool,
                  help="Whether to save all paths data to the snapshot")
    @autoargs.arg("plot", type=bool,
                  help="Plot evaluation run after each iteration")
    @autoargs.arg("pause_for_plot", type=bool,
                  help="Plot evaluation run after each iteration")
    @autoargs.arg("eta", type=float,
                  help="eta value for KL multiplier")
    @autoargs.arg("snn_n_samples", type=int,
                  help="snn_n_samples")
    @autoargs.arg("prior_sd", type=float,
                  help="prior_sd")
    @autoargs.arg("use_kl_ratio", type=bool,
                  help="use_kl_ratio")
    @autoargs.arg("kl_q_len", type=int,
                  help="kl_q_len")
    @autoargs.arg("reverse_update_kl", type=bool,
                  help="reverse_update_kl")
    @autoargs.arg("symbolic_prior_kl", type=bool,
                  help="symbolic_prior_kl")
    @autoargs.arg("use_reverse_kl_reg", type=bool,
                  help="use_reverse_kl_reg")
    @autoargs.arg("reverse_kl_reg_factor", type=float,
                  help="reverse_kl_reg_factor")
    @autoargs.arg("use_replay_pool", type=bool,
                  help="Use replay pool for dynamics model training.")
    @autoargs.arg("eta_discount", type=float,
                  help="eta_discount")
    def __init__(
            self,
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
            **kwargs
    ):
        super(RLAlgorithm, self).__init__()
        self.opt.n_itr = n_itr
        self.opt.start_itr = start_itr
        self.opt.batch_size = batch_size
        self.opt.max_path_length = max_path_length
        self.opt.discount = discount
        self.opt.gae_lambda = gae_lambda
        self.opt.plot = plot
        self.opt.pause_for_plot = pause_for_plot
        self.opt.whole_paths = whole_paths
        self.opt.center_adv = center_adv
        self.opt.positive_adv = positive_adv
        self.opt.record_states = record_states
        self.opt.store_paths = store_paths
        self.opt.algorithm_parallelized = algorithm_parallelized
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

    def start_worker(self, mdp, policy, baseline):
        parallel_sampler.populate_task(mdp, policy)
        parallel_sampler.run_map(worker_inject_baseline, baseline)
        if self.opt.plot:
            plotter.init_plot(mdp, policy)

    def shutdown_worker(self):
        pass

    def train(self, mdp, policy, baseline, **kwargs):

        ################
        ### SNN init ###
        ################
        batch_size = 1
        n_batches = 5  # temp

        obs_dim = np.sum(mdp.observation_shape)
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

        print("Building SNN model (eta={}) ...".format(self.eta))
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
                observation_shape=mdp.observation_shape,
                action_dim=1,  # mdp.action_dim,
            )

        # Start RL
        self.start_worker(mdp, policy, baseline)
        opt_info = self.init_opt(mdp, policy, baseline)
        for itr in xrange(self.opt.start_itr, self.opt.n_itr):
            logger.push_prefix('itr #%d | ' % itr)
            samples_data = self.obtain_samples(itr, mdp, policy, baseline)

            if self.use_replay_pool:
                # Fill replay pool.
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

#                     print('old_acc: {:.3f}'.format(old_acc))
#                     print('new_acc: {:.3f}'.format(new_acc))

            opt_info = self.optimize_policy(
                itr, policy, samples_data, opt_info)
            logger.log("saving snapshot...")
            params = self.get_itr_snapshot(
                itr, mdp, policy, baseline, samples_data, opt_info)
            if self.opt.store_paths:
                params["paths"] = samples_data["paths"]
            logger.save_itr_params(itr, params)
            logger.log("saved")
            logger.dump_tabular(with_prefix=False)
            logger.pop_prefix()
            if self.opt.plot:
                self.update_plot(policy)
                if self.opt.pause_for_plot:
                    raw_input("Plotting evaluation run: Press Enter to "
                              "continue...")
        self.shutdown_worker()

    def init_opt(self, mdp, policy, baseline):
        """
        Initialize the optimization procedure. If using theano / cgt, this may
        include declaring all the variables and compiling functions
        """
        raise NotImplementedError

    def get_itr_snapshot(self, itr, mdp, policy, baseline, samples_data,
                         opt_info):
        """
        Returns all the data that should be saved in the snapshot for this
        iteration.
        """
        raise NotImplementedError

    def optimize_policy(self, itr, policy, samples_data, opt_info):
        raise NotImplementedError

    def update_plot(self, policy):
        if self.opt.plot:
            plotter.update_plot(policy, self.opt.max_path_length)

    def obtain_samples(self, itr, mdp, policy, baseline):
        cur_params = policy.get_param_values()

        parallel_sampler.request_samples(
            policy_params=cur_params,
            max_samples=self.opt.batch_size,
            max_path_length=self.opt.max_path_length,
            whole_paths=self.opt.whole_paths,
            record_states=self.opt.record_states,
        )

        ####################
        ### UNN training ###
        ####################
        logger.log("fitting dynamics model...")
        # Compute sample Bellman error.
        paths = retrieve_paths()
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
            # FIXME: perhaps this is not the correct approach: we are updating
            # the model as if we are training on the total data set. However,
            # we are updating the model one sample at a time, given that the
            # model is already trained on previous data. Therefore, we should
            # replace the prior here with the current posterior.
            self.pnn.train_update_fn(_inputs[i][None, :], _targets[i][None, :])

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
        for i in xrange(len(G.paths)):
            path_length = len(G.paths[i]['rewards'])
            G.paths[i]['rewards_orig'] = np.array(G.paths[i]['rewards'])
            G.paths[i]['rewards'] = G.paths[i]['rewards'] + \
                kl[n_samples:n_samples + path_length]
            n_samples += path_length

        logger.log("fitted")

        parallel_sampler.run_map(worker_compute_paths_returns, self.opt)

        results = parallel_sampler.run_map(worker_process_paths, self.opt)

        logger.log("fitting baseline...")
        if baseline.algorithm_parallelized:
            try:
                baseline.fit()
            except Exception as e:
                import ipdb
                ipdb.set_trace()
        else:
            if self.opt.algorithm_parallelized:
                print "[Warning] Baseline should be parallelized when using a " \
                      "parallel algorithm for best possible performance"
            paths = retrieve_paths()
            baseline.fit(paths)
        parallel_sampler.run_map(
            worker_update_baseline,
            baseline.get_param_values(trainable=True)
        )
        logger.log("fitted")

        average_discounted_returns, average_returns, std_returns, max_returns, \
            min_returns, num_trajses, ents, evs = extract(
                results,
                "average_discounted_return", "average_return", "std_return",
                "max_return", "min_return", "num_trajs", "ent", "ev"
            )

        logger.record_tabular('Iteration', itr)
        logger.record_tabular('Entropy', np.mean(ents))
        logger.record_tabular('Perplexity', np.exp(np.mean(ents)))
        logger.record_tabular('AverageReturn',
                              np.mean(average_returns))
        logger.record_tabular('StdReturn',
                              np.mean(std_returns))
        logger.record_tabular('MaxReturn',
                              np.max(max_returns))
        logger.record_tabular('MinReturn',
                              np.min(min_returns))
        logger.record_tabular('AverageDiscountedReturn',
                              np.mean(average_discounted_returns))
        logger.record_tabular('NumTrajs', np.sum(num_trajses))
        logger.record_tabular('ExplainedVariance', np.mean(evs))
        logger.record_tabular('SNN_MeanKL', np.mean(kl))
        logger.record_tabular('SNN_StdKL', np.std(kl))
        logger.record_tabular('SNN_MinKL', np.min(kl))
        logger.record_tabular('SNN_MaxKL', np.max(kl))
        logger.record_tabular('SNN_DynModelSqLossBefore', old_acc)
        logger.record_tabular('SNN_DynModelSqLossAfter', new_acc)
        logger.record_tabular('eta', self.eta)

        mdp.log_extra()
        policy.log_extra()
        baseline.log_extra()

        if not self.opt.algorithm_parallelized:
            return aggregate_samples_data()
        else:
            return dict()

        # numerical check
        check_param = policy.get_params()
        if np.any(np.isnan(check_param)):
            raise ArithmeticError("NaN in params")
        elif np.any(np.isinf(check_param)):
            raise ArithmeticError("InF in params")
