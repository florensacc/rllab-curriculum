from rllab.algo.base import RLAlgorithm
from rllab.algo.util import ReplayPool
from rllab.algo.first_order_method import parse_update_method
from rllab.misc.overrides import overrides
from rllab.misc import autoargs
from rllab.misc.special import discount_return, discount_cumsum, \
    explained_variance_1d
from rllab.misc.ext import compile_function, new_tensor, merge_dict, extract
from rllab.sampler import parallel_sampler
from rllab.plotter import plotter
from collections import OrderedDict
import rllab.misc.logger as logger
import theano.tensor as TT
import cPickle as pickle
import numpy as np
import pyprind
import theano
import collections
import copy
from rllab.es.ou_strategy import OUStrategy


class ReplayPool(object):

    def __init__(self,
                 pool_size=1000
                 ):
        self.pool_observations = collections.deque(maxlen=pool_size)
        self.pool_rewards = collections.deque(maxlen=pool_size)
        self.pool_actions = collections.deque(maxlen=pool_size)

    def add_sample(self, observation, action, reward):
        self.pool_observations.append(observation)
        self.pool_rewards.append(reward)
        self.pool_actions.append(action)

    def get_size(self):
        return len(self.pool_observations)

    def get_random_batch(self, batch_size):
        # Never take last sample.
        r_ids = np.random.randint(0, len(self.pool) - 1, batch_size)
        r_obs = self.pool_observations[r_ids]
        r_rew = self.pool_rewards[r_ids]
        r_act = self.pool_actions[r_ids]
        r_obs_nxt = self.pool_observations[r_ids + 1]
        return r_obs, r_rew, r_act, r_obs_nxt


class DDPG(RLAlgorithm):
    """
    Very basic version of:
    Deep Deterministic Policy Gradient (DDPG) by Lillicrap (2015).
    """

    def __init__(self,
                 batch_size=64,
                 n_epochs=np.inf,
                 qf_learning_rate=1e-4,
                 policy_learning_rate=1e-4,
                 discount=0.99,
                 qf_update_method='sgd',
                 policy_update_method='sgd',
                 replay_pool_size=10000,
                 min_replay_pool_size=1000
                 ):
        self.batch_size = batch_size
        self.qf_learning_rate = qf_learning_rate
        self.policy_learning_rate = policy_learning_rate
        self.discount = discount
        self.qf_update_method = qf_update_method
        self.policy_update_method = policy_update_method
        self.replay_pool_size = replay_pool_size
        self.min_replay_pool_size = min_replay_pool_size

    def init_opt(self, mdp, policy, qf):
        obs = new_tensor(
            'obs',
            ndim=1 + len(mdp.observation_shape),
            dtype=mdp.observation_dtype
        )
        obs_nxt = new_tensor(
            'obs',
            ndim=1 + len(mdp.observation_shape),
            dtype=mdp.observation_dtype
        )
        act = TT.matrix('act', dtype=mdp.action_dtype)
        rew = TT.vector('rew')

        # Fixed-target Qf and policy.
        fixed_qf = copy.deepcopy(qf)
        fixed_policy = copy.deepcopy(policy)

        y = rew + self.discount * \
            fixed_qf.get_qval_sym(
                obs_nxt, fixed_policy.get_action_sym(obs_nxt))

        critic_loss = TT.mean(TT.square(y - qf.get_qval_sym(obs, act)))
        actor_loss = TT.mean(qf.get_qval_sym(obs, act))

        critic_updates = self.qf_update_method(
            critic_loss, qf.get_params(trainable=True))
        actor_updates = self.policy_update_method(
            actor_loss, policy.get_params(trainable=True))

        f_critic_updates = compile_function(
            input=[obs, act, rew, obs_nxt], output=None, updates=critic_updates)
        f_actor_updates = compile_function(
            input=[obs, act], output=None, updates=actor_updates)

        return dict(
            f_critic_updates=f_critic_updates,
            f_actor_updates=f_actor_updates
        )

    def train_minitbatch(self, opt_info, minibatch):
        pass

    def train(self, mdp, policy, qf, es, **kwargs):
        replay_pool = ReplayPool(self.replay_pool_size)

        self.start_worker(mdp, policy)
        opt_info = self.init_opt(mdp, policy, qf)
        state, observation = mdp.reset()
        es = OUStrategy(mdp, 0., 0.05, 0.05)
        path_ended = False
        returns = []
        _return = 0.

        for _ in xrange(self.n_epochs):
            # Sample step until minpoolsize is ok.

            if path_ended:
                # Reset exploration process.
                es = OUStrategy(mdp, 0., 0.05, 0.05)
                returns.append(_return)

            action = es.get_action()
            state_nxt, observation_nxt, reward, path_ended = mdp.step(
                state, action)
            # Add to replay pool.
            replay_pool.add_sample(observation, action, reward)
            _return += reward

            # Update current state.
            state = state_nxt
            observation = observation_nxt

            if replay_pool.get_size() >= self.min_replay_pool_size:
                # Sample random minibatch.
                minibatch = replay_pool.get_random_batch()
                # Train minibatch
                self.train_minitbatch(opt_info, minibatch)
