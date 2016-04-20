from __future__ import print_function
from __future__ import absolute_import

from rllab.algos.base import RLAlgorithm
from rllab.algos.ddpg import parse_update_method, SimpleReplayPool
from rllab.sampler import parallel_sampler
from rllab.plotter import plotter
from rllab.misc import logger
from rllab.misc import ext
from rllab.misc import special
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
import numpy as np
import pyprind
import cPickle as pickle
import theano
import theano.tensor as TT


class NAC(RLAlgorithm):
    """
    Natural Actor Critic. Form a surrogate function d/dtheta E[R(tau)] = E[d/dtheta Q^pi(s,a(s,ep))]. Policy is
    optimized using conjugate gradient.
    """

    def __init__(
            self,
            env,
            policy,
            qf,
            baseline,
            qf_batch_size=32,
            n_epochs=200,
            policy_update_interval=1000,
            policy_batch_size=10000,
            qf_update_itrs=1000,
            min_pool_size=10000,
            replay_pool_size=1000000,
            policy_optimizer=None,
            policy_optimizer_args=None,
            policy_step_size=0.01,
            discount=0.99,
            max_path_length=250,
            qf_weight_decay=0.,
            qf_update_method='adam',
            qf_learning_rate=1e-3,
            eval_samples=10000,
            soft_target_tau=0.001,
            n_updates_per_sample=1,
            scale_reward=1.0,
            include_horizon_terminal_transitions=False,
            plot=False):
        """
        :param env: Environment
        :param policy: Policy
        :param qf: Q function
        :param qf_batch_size: Number of samples for each minibatch, which is used to train the Q function.
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
        :param plot: Whether to visualize the policy performance after each eval_interval.
        :return:
        """
        self.env = env
        self.policy = policy
        self.qf = qf
        self.baseline = baseline
        self.qf_batch_size = qf_batch_size
        self.policy_batch_size = policy_batch_size
        self.n_epochs = n_epochs
        self.policy_update_interval = policy_update_interval
        self.qf_update_itrs = qf_update_itrs
        self.min_pool_size = max(min_pool_size, qf_batch_size * 2, policy_batch_size * 2)
        self.replay_pool_size = replay_pool_size
        self.n_train_samples = 0
        self.discount = discount
        self.max_path_length = max_path_length
        self.qf_weight_decay = qf_weight_decay
        self.qf_update_method = \
            parse_update_method(
                qf_update_method,
                learning_rate=qf_learning_rate,
            )
        self.qf_learning_rate = qf_learning_rate
        self.eval_samples = eval_samples
        self.soft_target_tau = soft_target_tau
        self.n_updates_per_sample = n_updates_per_sample
        self.include_horizon_terminal_transitions = include_horizon_terminal_transitions
        self.plot = plot

        self.qf_loss_averages = []
        self.q_averages = []
        self.y_averages = []
        self.paths = []
        self.es_path_returns = []
        self.paths_samples_cnt = 0

        self.scale_reward = scale_reward

        if policy_optimizer is None:
            if policy_optimizer_args is None:
                policy_optimizer_args = dict()
            policy_optimizer = ConjugateGradientOptimizer(**policy_optimizer_args)
        self.policy_optimizer = policy_optimizer
        self.policy_step_size = policy_step_size

        self.qf_opt_info = None
        self.policy_opt_info = None

    def start_worker(self):
        parallel_sampler.populate_task(self.env, self.policy)
        if self.plot:
            plotter.init_plot(self.env, self.policy)

    def train(self):
        # This seems like a rather sequential method
        pool = SimpleReplayPool(
            max_pool_size=self.replay_pool_size,
            observation_dim=self.env.observation_space.flat_dim,
            action_dim=self.env.action_space.flat_dim,
        )
        self.start_worker()

        self.init_opt()
        path_length = 0
        path_return = 0
        terminal = False
        observation = self.env.reset()

        for epoch in xrange(self.n_epochs):
            logger.push_prefix('epoch #%d | ' % epoch)
            logger.log("Collecting samples")
            for _ in pyprind.prog_bar(xrange(self.policy_update_interval)):
                # Execute policy
                if terminal:  # or path_length > self.max_path_length:
                    # Note that if the last time step ends an episode, the very
                    # last state and observation will be ignored and not added
                    # to the replay pool
                    observation = self.env.reset()
                    self.policy.reset()
                    self.es_path_returns.append(path_return)
                    path_length = 0
                    path_return = 0
                action, _ = self.policy.get_action(observation)
                next_observation, reward, terminal, _ = self.env.step(action)
                self.n_train_samples += 1
                path_length += 1
                path_return += reward

                if not terminal and path_length >= self.max_path_length:
                    terminal = True
                    # only include the terminal transition in this case if the flag was set
                    if self.include_horizon_terminal_transitions:
                        pool.add_sample(observation, action, reward * self.scale_reward, terminal)
                else:
                    pool.add_sample(observation, action, reward * self.scale_reward, terminal)

                observation = next_observation

            if pool.size >= self.min_pool_size:
                logger.log("Training Q function")
                for _ in pyprind.prog_bar(xrange(self.qf_update_itrs)):
                    qf_batch = pool.random_batch(self.qf_batch_size)
                    self.train_qf(qf_batch)
                logger.log("Training policy")
                policy_batch = pool.random_batch(self.policy_batch_size)
                self.train_policy(policy_batch)
                logger.log("Training finished")
                self.evaluate(epoch)
                params = self.get_epoch_snapshot(epoch)
                logger.save_itr_params(epoch, params)
            logger.dump_tabular(with_prefix=False)
            logger.pop_prefix()

    def init_opt(self):
        self.init_qf_opt()
        self.init_policy_opt()

    def init_qf_opt(self):
        # First, create "target" Q functions
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

        qf_updates = self.qf_update_method(
            qf_reg_loss, self.qf.get_params(trainable=True))

        f_train_qf = ext.compile_function(
            inputs=[yvar, obs, action],
            outputs=[qf_loss, qval],
            updates=qf_updates
        )

        self.qf_opt_info = dict(
            f_train_qf=f_train_qf,
            target_qf=target_qf,
        )

    def init_policy_opt(self):

        obs_var = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1,
        )
        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1,
        )
        dist = self.policy.distribution
        old_dist_info_vars = {
            k: ext.new_tensor(
                'old_%s' % k,
                ndim=2,
                dtype=theano.config.floatX
            ) for k in dist.dist_info_keys
            }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

        dist_info_vars = self.policy.dist_info_sym(obs_var, action_var)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        reparam_action_var = self.policy.get_reparam_action_sym(obs_var, action_var, old_dist_info_vars)

        q_var = self.qf.get_qval_sym(obs_var, reparam_action_var)
        # TODO: normalize q values
        # TODO: baseline
        mean_kl = TT.mean(kl)
        surr_loss = - TT.mean(q_var)

        input_list = [
                         obs_var,
                         action_var,
                     ] + old_dist_info_vars_list

        self.policy_optimizer.update_opt(
            loss=surr_loss,
            target=self.policy,
            leq_constraint=(mean_kl, self.policy_step_size),
            inputs=input_list,
            constraint_name="mean_kl"
        )

    def train_qf(self, batch):

        obs, actions, rewards, next_obs, terminals = ext.extract(
            batch,
            "observations", "actions", "rewards", "next_observations",
            "terminals"
        )

        # compute the on-policy y values
        target_qf = self.qf_opt_info["target_qf"]

        next_actions, _ = self.policy.get_actions(next_obs)
        next_qvals = target_qf.get_qval(next_obs, next_actions)

        ys = rewards + (1. - terminals) * self.discount * next_qvals

        f_train_qf = self.qf_opt_info["f_train_qf"]

        qf_loss, qval = f_train_qf(ys, obs, actions)

        target_qf.set_param_values(
            target_qf.get_param_values() * (1.0 - self.soft_target_tau) +
            self.qf.get_param_values() * self.soft_target_tau)

        self.qf_loss_averages.append(qf_loss)
        self.q_averages.append(qval)
        self.y_averages.append(ys)

    def train_policy(self, batch):
        obs = batch["observations"]
        actions, agent_infos = self.policy.get_actions(obs)
        dist = self.policy.distribution
        dist_infos = [agent_infos[k] for k in dist.dist_info_keys]
        all_input_values = [obs, actions] + dist_infos
        loss_before = self.policy_optimizer.loss(all_input_values)
        self.policy_optimizer.optimize(all_input_values)
        mean_kl = self.policy_optimizer.constraint_val(all_input_values)
        loss_after = self.policy_optimizer.loss(all_input_values)
        logger.record_tabular('LossBefore', loss_before)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('dLoss', loss_before - loss_after)

    def evaluate(self, epoch):
        logger.log("Collecting samples for evaluation")
        paths = parallel_sampler.sample_paths(
            policy_params=self.policy.get_param_values(),
            max_samples=self.eval_samples,
            max_path_length=self.max_path_length,
        )

        average_discounted_return = np.mean(
            [special.discount_return(path["rewards"], self.discount) for path in paths]
        )

        returns = [sum(path["rewards"]) for path in paths]

        all_qs = np.concatenate(self.q_averages)
        all_ys = np.concatenate(self.y_averages)

        average_q_loss = np.mean(self.qf_loss_averages)
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
        logger.record_tabular('NSamples', self.n_train_samples)
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

        self.q_averages = []
        self.y_averages = []
        self.es_path_returns = []

    def get_epoch_snapshot(self, epoch):
        return dict(
            env=self.env,
            epoch=epoch,
            qf=self.qf,
            policy=self.policy,
            baseline=self.baseline,
            target_qf=self.qf_opt_info["target_qf"],
        )
