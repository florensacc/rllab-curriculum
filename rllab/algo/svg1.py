from rllab.algo.base import RLAlgorithm
from rllab.algo.util import ReplayPool
from rllab.misc.overrides import overrides
from rllab.misc import autoargs
from rllab.misc.special import discount_return, discount_cumsum
from rllab.misc.ext import compile_function, new_tensor, merge_dict
from rllab.sampler import parallel_sampler
from rllab.plotter import plotter
import rllab.misc.logger as logger
import theano
import theano.tensor as TT
import cPickle as pickle
import numpy as np
import pyprind


class SVG1(RLAlgorithm):
    """
    Stochastic Value Gradient - SVG(1).
    """

    def __init__(self):
        pass

    def start_worker(self, mdp, policy):
        parallel_sampler.populate_task(mdp, policy)
        if self.plot:
            plotter.init_plot(mdp, policy)

    @overrides
    def train(self, mdp, policy, qf, model, **kwargs):
        # This seems like a rather sequential method
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
            for epoch_itr in pyprind.prog_bar(xrange(self.epoch_length)):
                # Execute policy
                if terminal:
                    # Note that if the last time step ends an episode, the very
                    # last state and observation will be ignored and not added
                    # to the replay pool
                    state, observation = mdp.reset()
                    policy.episode_reset()
                action, pdist = policy.get_action(observation)

                next_state, next_observation, reward, terminal = \
                    mdp.step(state, action)

                self.record_step(pool, state, observation, action, pdist,
                                 next_state, next_observation, reward,
                                 terminal)

                state, observation = next_state, next_observation

                if pool.size >= self.min_pool_size:
                    # Train policy
                    batch = pool.random_batch(self.batch_size)
                    opt_info = self.do_training(
                        itr, batch, qf, policy, model, opt_info)

                itr += 1
            logger.log("Training finished")
            if pool.size >= self.min_pool_size:
                opt_info = self.evaluate(epoch, qf, policy, model, opt_info)
                yield opt_info
                params = self.get_epoch_snapshot(
                    epoch, qf, policy, model, opt_info)
                logger.save_itr_params(epoch, params)
            logger.dump_tabular(with_prefix=False)
            logger.pop_prefix()

    def init_opt(self, mdp, policy, qf):

        return dict()

    def record_step(self, pool, state, observation, action, pdist,
                    next_state, next_observation, reward, terminal):
        pool.add_sample(observation, action, reward, terminal, extra=pdist)

    def do_training(self, itr, batch, qf, policy, model, opt_info):
        states, actions, rewards, next_states, terminal, pdists = batch
        pass

    def evaluate(self, epoch, qf, policy, model, opt_info):
        pass

    def get_epoch_snapshot(self, epoch, qf, policy, model, opt_info):
        return dict(
            epoch=epoch,
            qf=qf,
            policy=policy,
            model=model,
        )
