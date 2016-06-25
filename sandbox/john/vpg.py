import rllab.misc.logger as logger
from rllab.core.serializable import Serializable
from rllab.misc import special
import tensorflow as tf
from collections import defaultdict
import numpy as np
from sandbox.john import tf_util as U, common


class VPG(Serializable):
    def __init__(
        self,
        venv,
        policy,
        horizon=20,
        discount=0.99,
        gae_lambda=1.0,        
        vf_coeff=0.001,
        stepsize = 0.001,
    ):
        Serializable.quick_init(self, locals())

        self.venv = venv
        self.policy = policy
        self.horizon = horizon
        self.discount = discount
        self.gae_lambda = gae_lambda

        ac_space = venv.action_space
        ob_space = venv.observation_space
        batch_size = venv.num_envs

        # Build graph
        # ----------------------------------------        
        self.syms = defaultdict(list)
        pol_loss = 0
        vf_loss = 0
        total_ent = 0
        for t in xrange(horizon):
            ob = common.space2variable(ob_space, name = "ob%i"%t, prepend_shape=[batch_size])
            ac, logp, vpred, ent = policy(ob)
            self.syms["vpred"].append(vpred)
            vtarg = tf.placeholder(tf.float32, shape=[batch_size], name="vtarg%i"%t)
            self.syms["vtarg"].append(vtarg)
            adv = tf.placeholder(tf.float32, shape=[batch_size])
            pol_loss -= tf.reduce_sum(logp * adv)
            vf_loss += tf.reduce_sum(tf.square(vpred - vtarg))
            total_ent += tf.reduce_sum(ent)
            self.syms["ob"].append(ob)
            self.syms["ac"].append(ac)
            self.syms["adv"].append(adv)
            self.syms["new"].append(tf.placeholder(tf.int32, shape=[batch_size], name="new%i"%t))
        loss = vf_loss * vf_coeff + pol_loss
        avg_ent = total_ent / (horizon * batch_size) # create at least one var!!!
        update_op = tf.train.AdamOptimizer(stepsize).minimize(loss)
        with tf.control_dependencies([update_op]):
            self.dummy = tf.constant(0.0)        
        
        self.losses = [vf_loss, pol_loss, loss, avg_ent]
        self.loss_names = ["vf", "pol", "total", "ent"]
        U.initialize()
        self.filt = common.ZFilter()

    def train(self, n_updates):
        batch_size = self.venv.num_envs
        ob_raw = self.venv.reset()
        ob = self.filt(ob_raw)
        new = np.ones(batch_size, bool)
        total_timesteps = 0
        total_episodes = 0
        rets = np.zeros(batch_size, 'float32')
        for i_update in xrange(n_updates):
            episode_returns = []
            setup = U.SESSION.partial_run_setup(
                self.syms["ac"] + self.syms["vpred"] + self.losses + [self.dummy],
                self.syms["ob"] + self.syms["adv"] + self.syms["vtarg"], 
                )
            rews = np.zeros((self.horizon, batch_size), 'float32')
            vpreds = np.zeros((self.horizon+1, batch_size), 'float32')
            news = np.zeros((self.horizon, batch_size) , 'float32')
            for t in xrange(self.horizon):
                news[t] = new
                ac, vpreds[t] = U.SESSION.partial_run(setup, [self.syms["ac"][t], self.syms["vpred"][t]], feed_dict={self.syms["ob"][t] : ob})
                ob_raw, rews[t], new = self.venv.step(ac)
                ob = self.filt(ob_raw)
                rets += rews[t]
                if total_timesteps > 0:
                    for i_env in np.flatnonzero(new):
                        episode_returns.append(rets[i_env])
                        rets[i_env] = 0
                total_timesteps += 1

            deltas = rews + self.discount * vpreds[1:] - vpreds[:-1]
            deltas[-1] = 0 # XXX 
            advs = common.discounted_future_sum(deltas, news, self.gae_lambda * self.discount)
            # Note that we demean advantages by computing mean at each timestep
            standardized_advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            vtargs = vpreds[:-1] + advs
            losses = U.SESSION.partial_run(setup, self.losses + [self.dummy], feed_dict = dict(
                zip(self.syms["adv"], standardized_advs) +
                zip(self.syms["vtarg"], vtargs)))[:-1]
            losses_dict = dict(zip(self.loss_names, losses))
            ent = losses_dict["ent"]
            total_episodes += len(episode_returns)         
            logger.record_tabular("Iteration", i_update)
            logger.record_tabular("NumTrajs", len(episode_returns))
            ok = len(episode_returns) > 0
            logger.record_tabular("MinReturn", np.min(episode_returns) if ok else np.nan)
            logger.record_tabular("MaxReturn", np.max(episode_returns) if ok else np.nan)
            logger.record_tabular("MeanReturn", np.mean(episode_returns) if ok else np.nan)
            logger.record_tabular("StdReturn", np.std(episode_returns) if ok else np.nan)
            logger.record_tabular("Entropy", ent)
            logger.record_tabular("Perplexity", np.exp(ent))
            logger.record_tabular("ExplainedVariance", special.explained_variance_1d(vpreds[:-1].ravel(), vtargs.ravel()))
            logger.record_tabular("TotalTimesteps", total_timesteps)
            logger.record_tabular("TotalEpisodes", total_episodes)

            logger.dump_tabular(with_prefix=False, with_timestamp=False)

