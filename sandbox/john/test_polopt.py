from rllab.misc import special
from rllab.envs.gym_env import GymEnv
from rllab import spaces
import numpy as np
from online_vpg import OnlineVPG
import tf_util as U
import tensorflow as tf


if 0:
    from dummy_vec_env import DummyVecEnv as VecEnv
    env = GymEnv("CartPole-v0")
    ve = VecEnv(env, 4, 16, max_path_length=200)
    n_actions = env.action_space.n
    batch_size = ve.num_envs

    @U.module
    def policy(ob):
        h1 = U.tanh(U.dense(ob, 128))
        u = U.dense(h1, n_actions+1, weight_init=U.NormalizedColumns(.1))
        vpred = u[:,0]*10.0
        # ac = tf.py_func(cat_sample, [p], [tf.float32])[0]
        logits = u[:,1:]
        ac = tf.multinomial(logits, 1)[:,0]
        alllogp = tf.nn.log_softmax(logits)
        logp = U.fancy_slice_2d(alllogp, tf.range(batch_size), ac)
        ent = - U.sum(alllogp * tf.exp(alllogp), axis=1)
        return ac, logp, vpred, ent

    alg = OnlineVPG(ve, policy, horizon=50, stepsize=0.01)
    alg.train(1000)

elif 1:
    from rpc_vec_env import RpcVecEnv as VecEnv    
    env = GymEnv("Hopper-v1")
    ve = VecEnv(env, 4, 64, max_path_length=500)
    out_size = env.action_space.shape[0]
    batch_size = ve.num_envs

    @U.module
    def policy(ob):
        h1 = U.tanh(U.dense(ob, 32))
        h2 = U.tanh(U.dense(h1, 32))
        u = U.dense(h2, out_size+1, weight_init=U.NormalizedColumns(.1))
        vpred = u[:,0] * 10.0
        # ac = tf.py_func(cat_sample, [p], [tf.float32])[0]
        mean = u[:,1:]
        logstdev = U.Variable(value=np.zeros((1, out_size), 'float32'))
        stdev = U.exp(logstdev)
        z = tf.random_normal(shape=[batch_size, out_size])
        ac = z * stdev + mean
        z1 = (tf.stop_gradient(ac) - mean) / stdev
        logp = - U.sum(logstdev) - 0.5 * U.sum(U.square(z1), axis=1)
        ent = U.mean(stdev)*batch_size # more interpretable than differential entropy
        return ac, logp, vpred, ent

    alg = OnlineVPG(ve, policy, horizon=50, discount=0.995, gae_lambda=0.99, stepsize=0.01)
    alg.train(1000)

