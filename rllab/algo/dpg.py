from rllab.algo.base import RLAlgorithm
from rllab.algo.util import ReplayPool
from rllab.algo.first_order_method import parse_update_method
from rllab.misc.overrides import overrides
from rllab.misc import autoargs
from rllab.misc.special import discount_return, discount_cumsum
from rllab.misc.ext import compile_function, new_tensor, merge_dict, extract
from rllab.sampler import parallel_sampler
from rllab.plotter import plotter
import rllab.misc.logger as logger
import theano
import theano.tensor as TT
import cPickle as pickle
import numpy as np
import pyprind


class DPG(RLAlgorithm):
    """
    Deterministic Policy Gradient.
    """

    @autoargs.arg('batch_size', type=int,
                  help='Number of samples for each minibatch.')
    @autoargs.arg('n_epochs', type=int,
                  help='Number of epochs. Policy will be evaluated after each '
                       'epoch.')
    @autoargs.arg('epoch_length', type=int,
                  help='How many timesteps for each epoch.')
    @autoargs.arg('min_pool_size', type=int,
                  help='Minimum size of the pool to start training.')
    @autoargs.arg('replay_pool_size', type=int,
                  help='Size of the experience replay pool.')
    @autoargs.arg('discount', type=float,
                  help='Discount factor for the cumulative return.')
    @autoargs.arg('max_path_length', type=int,
                  help='Discount factor for the cumulative return.')
    @autoargs.arg('qf_weight_decay', type=float,
                  help='Weight decay factor for parameters of the Q function.')
    @autoargs.arg('qf_update_method', type=str,
                  help='Online optimization method for training Q function.')
    @autoargs.arg('qf_learning_rate', type=float,
                  help='Learning rate for training Q function.')
    @autoargs.arg('policy_weight_decay', type=float,
                  help='Weight decay factor for parameters of the policy.')
    @autoargs.arg('policy_update_method', type=str,
                  help='Online optimization method for training the policy.')
    @autoargs.arg('policy_learning_rate', type=float,
                  help='Learning rate for training the policy.')
    @autoargs.arg('eval_samples', type=int,
                  help='Number of samples (timesteps) for evaluating the '
                       'policy.')
    @autoargs.arg('eval_whole_paths', type=bool,
                  help='Whether to make sure that all trajectories are '
                       'executed until the terminal state or the '
                       'max_path_length, even at the expense of possibly more '
                       'samples for evaluation.')
    @autoargs.arg('soft_target_tau', type=float,
                  help='Interpolation parameter for doing the soft target '
                       'update.')
    @autoargs.arg('plot', type=bool,
                  help='Whether to visualize the policy performance after '
                       'each eval_interval.')
    def __init__(
            self,
            batch_size=64,
            n_epochs=200,
            epoch_length=10000,
            min_pool_size=10000,
            replay_pool_size=1000000,
            discount=0.99,
            max_path_length=500,
            qf_weight_decay=0.01,
            qf_update_method='adam',
            qf_learning_rate=1e-4,
            policy_weight_decay=0,
            policy_update_method='adam',
            policy_learning_rate=1e-3,
            eval_samples=10000,
            eval_whole_paths=True,
            soft_target_tau=0.001,
            plot=False):
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
                learning_rate=qf_learning_rate
            )
        self.policy_weight_decay = policy_weight_decay
        self.policy_update_method = \
            parse_update_method(
                policy_update_method,
                learning_rate=policy_learning_rate
            )
        self.eval_samples = eval_samples
        self.eval_whole_paths = eval_whole_paths
        self.soft_target_tau = soft_target_tau
        self.plot = plot

        self.qf_loss_averages = []
        self.policy_surr_averages = []
        self.q_averages = []
        self.paths = []
        self.paths_samples_cnt = 0

    def start_worker(self, mdp, policy):
        parallel_sampler.populate_task(mdp, policy)
        if self.plot:
            plotter.init_plot(mdp, policy)

    @overrides
    def train(self, mdp, policy, qf, es, **kwargs):
        # This seems like a rather sequential method
        terminal = True
        pool = ReplayPool(
            state_shape=mdp.observation_shape,
            action_dim=mdp.action_dim,
            max_steps=self.replay_pool_size,
        )
        self.start_worker(mdp, policy)
        opt_info = self.init_opt(mdp, policy, qf)
        itr = 0
        for epoch in xrange(self.n_epochs):
            logger.push_prefix('epoch #%d | ' % epoch)
            logger.log("Training started")
            for epoch_itr in pyprind.prog_bar(xrange(self.epoch_length)):
                # Execute policy
                if terminal:
                    # Note that if the last time step ends an episode, the very
                    # last state and observation will be ignored and not added
                    # to the replay pool
                    state, observation = mdp.reset()
                    es.episode_reset()
                action = es.get_action(itr, observation, policy=policy, qf=qf)

                next_state, next_observation, reward, terminal = \
                    mdp.step(state, action)

                pool.add_sample(observation, action, reward, terminal)
                state, observation = next_state, next_observation

                if pool.size >= self.min_pool_size:
                    # Train policy
                    batch = pool.random_batch(self.batch_size)
                    opt_info = self.do_training(
                        itr, batch, qf, policy, opt_info)

                itr += 1
            logger.log("Training finished")
            if pool.size >= self.min_pool_size:
                opt_info = self.evaluate(epoch, qf, policy, opt_info)
                yield opt_info
                params = self.get_epoch_snapshot(
                    epoch, qf, policy, es, opt_info)
                logger.save_itr_params(epoch, params)
            logger.dump_tabular(with_prefix=False)
            logger.pop_prefix()

    def init_opt(self, mdp, policy, qf):

        # First, create "target" policy and Q functions
        target_policy = pickle.loads(pickle.dumps(policy))
        target_qf = pickle.loads(pickle.dumps(qf))

        # y need to be computed first
        obs = new_tensor(
            'obs',
            ndim=1+len(mdp.observation_shape),
            dtype=mdp.observation_dtype
        )
        next_obs = new_tensor(
            'next_obs',
            ndim=1+len(mdp.observation_shape),
            dtype=mdp.observation_dtype
        )

        rewards = TT.vector('rewards')
        terminals = TT.vector('terminals')

        # compute the on-policy y values
        next_qval = target_qf.get_qval_sym(
            next_obs,
            target_policy.get_action_sym(next_obs, train=True),
            train=True
        )

        ys = rewards + (1 - terminals) * self.discount * next_qval
        f_y = compile_function(
            inputs=[next_obs, rewards, terminals],
            outputs=ys
        )

        # The yi values are computed separately as above and then passed to
        # the training functions below
        action = TT.matrix('action', dtype=mdp.action_dtype)
        yvar = TT.vector('ys')

        qf_weight_decay_term = self.qf_weight_decay * \
            sum([TT.sum(TT.square(param)) for param in qf.params])

        qval = qf.get_qval_sym(obs, action, train=True)
        diff = TT.square(qf.normalize_sym(qval) -
                         qf.normalize_sym(yvar))
        qf_loss = TT.mean(diff)
        qf_reg_loss = qf_loss + qf_weight_decay_term

        policy_weight_decay_term = self.policy_weight_decay * \
            sum([TT.sum(TT.square(param)) for param in policy.params])
        policy_qval = qf.get_qval_sym(
            obs, policy.get_action_sym(obs, train=True), train=True)
        # The policy gradient is computed with respect to the unscaled Q values
        policy_surr = -TT.mean(qf.normalize_sym(policy_qval))
        policy_reg_surr = policy_surr + policy_weight_decay_term

        qf_updates = merge_dict(
            self.qf_update_method(qf_reg_loss, qf.params),
            qf.get_default_updates(obs, action, train=True)
        )

        policy_updates = merge_dict(
            self.policy_update_method(policy_reg_surr, policy.params),
            policy.get_default_updates(obs)
        )

        f_normalize_qf = compile_function(
            inputs=[yvar],
            updates=qf.normalize_updates(yvar),
        )

        f_train_qf = compile_function(
            inputs=[yvar, obs, action, rewards],
            outputs=[qf_loss, qval],
            updates=qf_updates
        )
        f_train_policy = compile_function(
            inputs=[yvar, obs],
            outputs=policy_surr,
            updates=policy_updates
        )

        return dict(
            f_y=f_y,
            f_normalize_qf=f_normalize_qf,
            f_train_qf=f_train_qf,
            f_train_policy=f_train_policy,
            target_qf=target_qf,
            target_policy=target_policy,
        )

    def do_training(self, itr, batch, qf, policy, opt_info):

        states, actions, rewards, next_states, terminal = extract(
            batch,
            "states", "actions", "rewards", "next_states", "terminals"
        )

        f_y = opt_info["f_y"]
        f_normalize_qf = opt_info["f_normalize_qf"]
        f_train_qf = opt_info["f_train_qf"]
        f_train_policy = opt_info["f_train_policy"]
        target_qf = opt_info["target_qf"]
        target_policy = opt_info["target_policy"]

        ys = f_y(next_states, rewards, terminal)
        f_normalize_qf(ys)
        qf_loss, qval = f_train_qf(ys, states, actions, rewards)
        policy_surr = f_train_policy(ys, states)

        self.qf_loss_averages.append(qf_loss)
        self.policy_surr_averages.append(policy_surr)
        self.q_averages.append(qval)

        target_qf.set_param_values(
            self.soft_target_tau * qf.get_param_values() +
            (1 - self.soft_target_tau) * target_qf.get_param_values()
        )
        target_policy.set_param_values(
            self.soft_target_tau * policy.get_param_values() +
            (1 - self.soft_target_tau) * target_policy.get_param_values()
        )

        return opt_info

    def evaluate(self, epoch, qf, policy, opt_info):
        logger.log("Collecting samples for evaluation")

        paths = parallel_sampler.request_samples(
            policy_params=policy.get_param_values(),
            max_samples=self.eval_samples,
            max_path_length=self.max_path_length,
            whole_paths=self.eval_whole_paths,
        )
        average_discounted_return = np.mean(
            [discount_return(path["rewards"], self.discount) for path in paths]
        )

        returns = [sum(path["rewards"]) for path in paths]

        average_q_loss = np.mean(self.qf_loss_averages)
        average_policy_surr = np.mean(self.policy_surr_averages)
        average_q = np.mean(np.concatenate(self.q_averages))
        average_action = np.mean(np.square(np.concatenate(
            [path["actions"] for path in paths]
        )))

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
        logger.record_tabular('AverageQLoss', average_q_loss)
        logger.record_tabular('AveragePolicySurr', average_policy_surr)
        logger.record_tabular('AverageQ', average_q)
        logger.record_tabular('AverageAction', average_action)
        logger.record_tabular('PolicyParamNorm',
                              np.linalg.norm(policy.get_param_values()))
        logger.record_tabular('QFunParamNorm',
                              np.linalg.norm(qf.get_param_values()))

        self.qf_loss_averages = []
        self.policy_surr_averages = []
        self.q_averages = []

        return merge_dict(opt_info, dict(
            eval_paths=paths,
        ))

    def get_epoch_snapshot(self, epoch, qf, policy, es, opt_info):
        return dict(
            epoch=epoch,
            qf=qf,
            policy=policy,
            es=es,
        )
