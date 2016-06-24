import rllab.misc.logger as logger
from rllab.core.serializable import Serializable
from rllab import spaces
from rllab.misc import special
import tensorflow as tf
import tf_util as U
from collections import defaultdict
import numpy as np

def make_batch_variable(space, batch_size = None, name = None):
    if isinstance(space, spaces.Box):        
        return tf.placeholder(tf.float32, shape=[batch_size] + [space.flat_dim], name=name)
    elif isinstance(space, spaces.Discrete):
        return tf.placeholder(tf.int32, shape=[batch_size] + [space.flat_dim], name=name)
    else:
        raise NotImplementedError

class OnlineVPG(Serializable):
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
        self.syms = defaultdict(list)
        batch_size = venv.num_envs
        pol_loss = 0
        vf_loss = 0
        total_ent = 0
        for t in xrange(horizon):
            ob = make_batch_variable(ob_space, name = "ob%i"%t)
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
        update_op = tf.train.AdamOptimizer(stepsize).minimize(loss)
        with tf.control_dependencies([update_op]):
            avg_ent = total_ent / (horizon * batch_size) # create at least one var!!!
        self.losses_after = [vf_loss, pol_loss, loss, avg_ent]
        self.loss_names = ["vf", "pol", "total", "ent"]
        U.initialize()

    def train(self, n_updates):
        batch_size = self.venv.num_envs
        ob_raw = self.venv.reset()
        new = np.ones(batch_size, bool)
        ob_mean = ob_raw.mean(axis=0, keepdims=True)
        ob_scale = ob_raw.std(axis=0, keepdims=True)
        count = 1
        rets = np.zeros(batch_size, 'float32')
        for i_update in xrange(n_updates):
            total_rets = []
            setup = U.SESSION.partial_run_setup(
                self.syms["ac"] + self.syms["vpred"] + self.losses_after,
                self.syms["ob"] + self.syms["adv"] + self.syms["vtarg"], 
                )
            rews = np.zeros((self.horizon, batch_size), 'float32')
            vpreds = np.zeros((self.horizon+1, batch_size), 'float32')
            news = np.zeros((self.horizon, batch_size) , 'float32')
            for t in xrange(self.horizon):
                ob = np.clip((ob_raw - ob_mean) / ob_scale, -10.0, 10.0)
                news[t] = new
                ac, vpreds[t] = U.SESSION.partial_run(setup, [self.syms["ac"][t], self.syms["vpred"][t]], feed_dict={self.syms["ob"][t] : ob})
                ob_raw, rews[t], new = self.venv.step(ac)
                rets += rews[t]
                if count > 0:
                    for i_env in np.flatnonzero(new):
                        total_rets.append(rets[i_env])
                        rets[i_env] = 0
                # Filter update:
                ob_mean = (ob_mean * count + ob_raw.mean(axis=0, keepdims=True)) / (count + 1)
                ob_scale = np.sqrt((np.square(ob_scale) * count + np.square(ob_mean - ob_raw).mean(axis=0, keepdims=True) ) / (count + 1))
                count += 1

            deltas = rews + self.discount * vpreds[1:] - vpreds[:-1]
            deltas[-1] = 0
            advs = np.zeros_like(deltas)
            q = self.gae_lambda * self.discount
            advs[-1] = deltas[-1]            
            for t1 in xrange(self.horizon - 2, -1, -1):
                advs[t1] += deltas[t1] + advs[t1 + 1] * q * (1 - news[t1 + 1])
            standardized_advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            vtargs = vpreds[:-1] + advs
            losses = U.SESSION.partial_run(setup, self.losses_after, feed_dict = dict(
                zip(self.syms["adv"], standardized_advs) +
                zip(self.syms["vtarg"], vtargs)))
            losses_dict = dict(zip(self.loss_names, losses))
            ent = losses_dict["ent"]
            logger.record_tabular("Iteration", i_update)
            logger.record_tabular("NumTrajs", len(total_rets))
            logger.record_tabular("MinReturn", np.min(total_rets))
            logger.record_tabular("MaxReturn", np.max(total_rets))
            logger.record_tabular("MeanReturn", np.mean(total_rets))
            logger.record_tabular("StdReturn", np.std(total_rets))
            logger.record_tabular("Entropy", ent)
            logger.record_tabular("Perplexity", np.exp(ent))
            logger.record_tabular("ExplainedVariance", special.explained_variance_1d(vpreds[:-1].ravel(), vtargs.ravel()))


            logger.dump_tabular(with_prefix=False)

