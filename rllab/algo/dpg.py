from rllab.algo.base import RLAlgorithm
from rllab.algo.util import ReplayPool
from rllab.algo.first_order_method import parse_update_method
from rllab.misc.overrides import overrides
from rllab.misc import autoargs
from rllab.misc.ext import compile_function
import tensorfuse.tensor as TT
import cPickle as pickle
from rllab.sampler import parallel_sampler
from rllab.plotter import plotter


class DPG(RLAlgorithm):
    """
    Deterministic Policy Gradient.
    """

    @autoargs.arg('batch_size', type=int,
                  help='Number of samples for each minibatch')
    @autoargs.arg('n_itr', type=int,
                  help='Number of timesteps')
    @autoargs.arg('min_pool_size', type=int,
                  help='Minimum size of the pool to start training')
    def __init__(
            self,
            batch_size=64,
            n_itr=int(1e7),
            min_pool_size=10000,
            replay_pool_size=int(1e6),
            discount=0.99,
            qfun_weight_decay=0.01,
            qfun_update_method='adam',
            qfun_learning_rate=1e-4,
            policy_update_method='adam',
            policy_learning_rate=1e-3,
            eval_interval=1000,
            soft_target_tau=0.001,
            plot=False):
        self.batch_size = batch_size
        self.n_itr = n_itr
        self.min_pool_size = min_pool_size
        self.replay_pool_size = replay_pool_size
        self.discount = discount
        self.qfun_weight_decay = qfun_weight_decay
        self.qfun_update_method = \
            parse_update_method(
                qfun_update_method,
                learning_rate=qfun_learning_rate
            )
        self.policy_update_method = \
            parse_update_method(
                policy_update_method,
                learning_rate=policy_learning_rate
            )
        self.eval_interval = eval_interval
        self.soft_target_tau = soft_target_tau
        self.plot = plot

    def start_worker(self, mdp, policy):
        parallel_sampler.populate_task(mdp, policy)
        if self.plot:
            plotter.init_plot(mdp, policy)

    @overrides
    def train(self, mdp, policy, qfun, vf, explore_strategy, **kwargs):
        # This seems like a rather sequential method
        terminal = True
        pool = ReplayPool(
            state_shape=mdp.observation_shape,
            action_dim=mdp.action_dim,
            max_steps=self.replay_pool_size
        )
        self.start_worker(mdp, policy)
        opt_info = self.init_opt(mdp, policy, qfun, vf)
        for itr in xrange(self.n_itr):
            # Execute policy
            if terminal:
                # Note that if the last time step ends an episode, the very
                # last state and observation will be ignored and not added to
                # the replay pool
                state, observation = mdp.reset()
                explore_strategy.episode_reset()
            action = explore_strategy.get_action(itr, observation, policy, mdp)
            next_state, next_observation, reward, terminal = \
                mdp.step(state, action)
            pool.add_sample(observation, action, reward, terminal)
            state, observation = next_state, next_observation

            if pool.size >= self.min_pool_size:
                # Train policy
                batch = pool.random_batch(self.batch_size)
                opt_info = self.do_training(itr, batch, qfun, policy, opt_info)

                if itr > 0 and itr % self.eval_interval == 0:
                    opt_info = self.evaluate(
                        itr, batch, qfun, policy, opt_info)

    def init_opt(self, mdp, policy, qfun, vf):
        # First, create "target" policy and Q functions
        target_policy = pickle.loads(pickle.dumps(policy))
        target_qfun = pickle.loads(pickle.dumps(qfun))

        # y need to be computed first

        rewards = TT.vector('rewards')
        terminals = TT.vector('terminals')

        # obs = TT.tensor('observations', ndim=len(mdp.observation_shape)+1)

        # compute the on-policy y values
        ys = rewards + (1 - terminals) * self.discount * \
            target_qfun.get_policy_qvals(target_policy.actions_var)
        f_y = compile_function(
            inputs=[target_qfun.input_var, target_policy.input_var, rewards,
                    terminals],
            outputs=ys
        )

        # The yi values are computed separately as above and then passed to
        # the training functions below
        actions = TT.matrix('actions', dtype=mdp.action_dtype)
        yvar = TT.vector('ys')
        qfun_weight_decay_term = self.qfun_weight_decay * \
            sum([TT.sum(TT.square(param)) for param in qfun.params])
        qfun_loss = TT.mean(TT.square(yvar - qfun.get_qvals(actions))) + \
            qfun_weight_decay_term
        policy_surr = TT.mean(qfun.get_policy_qvals(policy.actions_var))

        qfun_updates = self.qfun_update_method(qfun_loss, qfun.params)
        policy_updates = self.policy_update_method(policy_surr, policy.params)

        f_train_qfun = compile_function(
            inputs=[yvar, qfun.input_var, actions, rewards],
            outputs=None,
            updates=qfun_updates
        )
        f_train_policy = compile_function(
            inputs=[yvar, qfun.input_var, policy.input_var, actions, rewards],
            outputs=None,
            updates=policy_updates
        )

        return dict(
            f_y=f_y,
            f_train_qfun=f_train_qfun,
            f_train_policy=f_train_policy,
            target_qfun=target_qfun,
            target_policy=target_policy,
        )

    def do_training(self, itr, batch, qfun, policy, opt_info):
        states, actions, rewards, next_states, terminal = batch

        f_y = opt_info["f_y"]
        f_train_qfun = opt_info["f_train_qfun"]
        f_train_policy = opt_info["f_train_policy"]
        target_qfun = opt_info["target_qfun"]
        target_policy = opt_info["target_policy"]

        ys = f_y(next_states, next_states, rewards, terminal)
        f_train_qfun(ys, states, actions, rewards)
        f_train_policy(ys, states, states, actions, rewards)

        target_qfun.set_param_values(
            self.soft_target_tau * qfun.get_param_values() +
            (1 - self.soft_target_tau) * target_qfun.get_param_values()
        )
        target_policy.set_param_values(
            self.soft_target_tau * policy.get_param_values() +
            (1 - self.soft_target_tau) * target_policy.get_param_values()
        )
        return opt_info

    def evaluate(self, itr, batch, qfun, policy, opt_info):
        parallel_sampler.request_samples(
            policy.get_param_values(),

