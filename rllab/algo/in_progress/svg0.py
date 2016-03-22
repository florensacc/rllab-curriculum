from rllab.algo.base import RLAlgorithm
from rllab.algo.util import ReplayPool
from rllab.algo.first_order_method import parse_update_method
from rllab.misc.overrides import overrides
from rllab.misc import autoargs
from rllab.misc.special import discount_return
from rllab.misc.ext import compile_function, new_tensor, extract
from rllab.sampler import parallel_sampler
from rllab.plotter import plotter
import cPickle as pickle
import rllab.misc.logger as logger
import theano.tensor as TT
import numpy as np
import pyprind


class SVG0(RLAlgorithm):
    """
    Model-Free Stochastic Value Gradient - SVG(0).
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
    @autoargs.arg('qf_update_method', type=str,
                  help='Online optimization method for training the Q '
                       'function.')
    @autoargs.arg('qf_learning_rate', type=float,
                  help='Learning rate for training the Q function.')
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
            qf_update_method='adam',
            qf_learning_rate=1e-4,
            policy_update_method='adam',
            policy_learning_rate=1e-4,
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
        self.qf_update_method = parse_update_method(
            qf_update_method,
            learning_rate=qf_learning_rate
        )
        self.policy_update_method = parse_update_method(
            policy_update_method,
            learning_rate=policy_learning_rate
        )
        self.eval_samples = eval_samples
        self.eval_whole_paths = eval_whole_paths
        self.soft_target_tau = soft_target_tau
        self.plot = plot

        self._qf_losses = []
        self._policy_objs = []

    def start_worker(self, mdp, policy):
        parallel_sampler.populate_task(mdp, policy)
        if self.plot:
            plotter.init_plot(mdp, policy)

    @overrides
    def train(self, mdp, policy, qf, **kwargs):
        # hack to reset at the beginning of training
        terminal = True
        pool = ReplayPool(
            state_shape=mdp.observation_shape,
            action_dim=mdp.action_dim,
            max_steps=self.replay_pool_size
        )
        self.start_worker(mdp, policy)
        opt_info = self.init_opt(mdp, policy, qf)
        itr = 0
        for epoch in xrange(self.n_epochs):
            logger.push_prefix('epoch #%d | ' % epoch)
            logger.log("Training started")
            for itr in pyprind.prog_bar(xrange(itr, itr + self.epoch_length)):
                # Execute policy
                if terminal:
                    # Note that if the last time step ends an episode, the very
                    # last state and observation will be ignored and not added
                    # to the replay pool
                    state, observation = mdp.reset()
                    policy.reset()
                action, pdist = policy.act(observation)

                next_state, next_observation, reward, terminal = \
                    mdp.step(state, action)

                self.record_step(pool, state, observation, action, pdist,
                                 next_state, next_observation, reward,
                                 terminal)

                state, observation = next_state, next_observation

                if pool.size >= self.min_pool_size:
                    batch = pool.random_batch(self.batch_size)
                    opt_info = self.do_training(
                        itr, batch, qf, policy, opt_info)

                # itr += 1
            logger.log("Training finished")
            if pool.size >= self.min_pool_size:
                opt_info = self.evaluate(
                    epoch, mdp, qf, policy, opt_info)
                yield opt_info
                params = self.get_epoch_snapshot(
                    epoch, qf, policy, opt_info)
                logger.save_itr_params(epoch, params)
            logger.dump_tabular(with_prefix=False)
            logger.pop_prefix()

    def init_opt(self, mdp, policy, qf):

        # First, create "target" policy and Q functions
        target_policy = pickle.loads(pickle.dumps(policy))
        target_qf = pickle.loads(pickle.dumps(qf))

        obs = new_tensor(
            'obs',
            ndim=1+len(mdp.observation_shape),
            dtype=mdp.observation_dtype
        )
        actions = TT.matrix('actions', dtype=mdp.action_dtype)
        next_obs = new_tensor(
            'next_obs',
            ndim=1+len(mdp.observation_shape),
            dtype=mdp.observation_dtype
        )
        next_etas = TT.matrix('next_etas')

        terminals = TT.vector('terminals')
        old_pdists = TT.matrix('old_pdists')

        rewards = TT.vector('rewards')

        # Compute on-policy q function regression target
        ys = rewards + (1 - terminals) * self.discount * \
            target_qf.get_qval_sym(
                next_obs,
                target_policy.get_reparam_action_sym(next_obs, next_etas)
            )
        f_ys = compile_function(
            inputs=[rewards, terminals, next_obs, next_etas],
            outputs=ys,
        )

        # Train value function
        yvals = TT.vector('yvals')
        # pylint: disable=assignment-from-no-return
        diff = TT.square(qf.normalize_sym(qf.get_qval_sym(obs, actions)) -
                         qf.normalize_sym(yvals))
        # pylint: enable=assignment-from-no-return
        pdists = policy.get_pdist_sym(obs)
        weights = policy.likelihood_ratio(old_pdists, pdists, actions)
        f_weights = compile_function(
            inputs=[obs, actions, old_pdists],
            outputs=weights,
        )

        weight_vals = TT.vector('weights')
        qf_loss = TT.sum(weight_vals * diff) / TT.sum(weight_vals)

        f_normalize_qf = compile_function(
            inputs=[yvals],
            updates=qf.normalize_updates(yvals),
        )

        f_train_qf = compile_function(
            inputs=[obs, actions, old_pdists, yvals, weight_vals],
            outputs=qf_loss,
            updates=self.qf_update_method(qf_loss, qf.trainable_params),
        )

        # Train policy
        etas = TT.matrix('etas')
        reparam_actions = policy.get_reparam_action_sym(obs, etas)
        predicted_ys = qf.normalize_sym(
            qf.get_qval_sym(obs, reparam_actions)
        )
        # Negative sign since we are maximizing
        policy_obj = - TT.mean(predicted_ys)
        policy_updates = self.policy_update_method(
            policy_obj, policy.trainable_params)

        f_train_policy = compile_function(
            inputs=[obs, etas, actions, old_pdists, terminals, weight_vals],
            outputs=policy_obj,
            updates=policy_updates
        )

        return dict(
            f_ys=f_ys,
            f_weights=f_weights,
            f_normalize_qf=f_normalize_qf,
            f_train_qf=f_train_qf,
            f_train_policy=f_train_policy,
            target_qf=target_qf,
            target_policy=target_policy,
        )

    def record_step(self, pool, state, observation, action, pdist,
                    next_state, next_observation, reward, terminal):
        pool.add_sample(observation, action, reward, terminal, extra=pdist)

    def do_training(self, itr, batch, qf, policy, opt_info):
        obs, actions, rewards, next_obs, next_actions, terminals, pdists, \
            next_pdists = extract(
                batch,
                "states", "actions", "rewards", "next_states", "next_actions",
                "terminals", "extras", "next_extras"
            )
        etas = policy.infer_eta(pdists, actions)
        next_etas = policy.infer_eta(next_pdists, next_actions)

        f_ys = opt_info["f_ys"]
        yvals = f_ys(rewards, terminals, next_obs, next_etas)

        f_weights = opt_info["f_weights"]
        weights = f_weights(obs, actions, pdists)

        f_normalize_qf = opt_info["f_normalize_qf"]
        f_normalize_qf(yvals)

        f_train_qf = opt_info["f_train_qf"]
        qf_loss = f_train_qf(obs, actions, pdists, yvals, weights)
        self._qf_losses.append(qf_loss)

        f_train_policy = opt_info["f_train_policy"]
        policy_obj = f_train_policy(
            obs, etas, actions, pdists, terminals, weights)
        self._policy_objs.append(policy_obj)

        target_qf = opt_info["target_qf"]
        target_policy = opt_info["target_policy"]
        target_qf.set_param_values(
            self.soft_target_tau * qf.get_param_values() +
            (1 - self.soft_target_tau) * target_qf.get_param_values()
        )
        target_policy.set_param_values(
            self.soft_target_tau * policy.get_param_values() +
            (1 - self.soft_target_tau) * target_policy.get_param_values()
        )

        return opt_info

    def evaluate(self, epoch, mdp, qf, policy, opt_info):
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

        average_action = np.mean(np.square(np.concatenate(
            [path["actions"] for path in paths]
        )))

        pdists = np.vstack([path["pdists"] for path in paths])
        ent = policy.entropy(pdists)

        logger.record_tabular('Epoch', epoch)
        logger.record_tabular('Entropy', ent)
        logger.record_tabular('Perplexity', np.exp(ent))
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
        logger.record_tabular('AverageAction', average_action)
        logger.record_tabular('AverageQFunctionLoss',
                              np.mean(self._qf_losses))
        logger.record_tabular('AveragePolicyObjective',
                              np.mean(self._policy_objs))
        logger.record_tabular('QFunctionParamNorm',
                              np.linalg.norm(
                                  qf.get_trainable_param_values()))
        logger.record_tabular('PolicyParamNorm',
                              np.linalg.norm(
                                  policy.get_trainable_param_values()))
        mdp.log_extra(logger, paths)
        qf.log_extra(logger, paths)
        policy.log_extra(logger, paths)
        self._qf_losses = []
        self._policy_objs = []
        return opt_info

    def get_epoch_snapshot(self, epoch, qf, policy, opt_info):
        return dict(
            epoch=epoch,
            qf=qf,
            policy=policy,
        )
