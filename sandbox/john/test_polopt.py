# from rpc_vec_env import RpcVecEnv as VecEnv
from dummy_vec_env import DummyVecEnv as VecEnv
from rllab.misc import special
from rllab.envs.gym_env import GymEnv
import numpy as np
from online_vpg import OnlineVPG
import tf_util as U
import tensorflow as tf
env = GymEnv("CartPole-v0")
ve = VecEnv(env, 4, 16, max_path_length=200)
n_actions = env.action_space.n
batch_size = ve.num_envs
@U.module
def policy(ob):
    h1 = U.tanh(U.dense(ob, 128))
    u = U.dense(h1, n_actions+1, weight_init=U.NormalizedColumns(.1))
    vpred = u[:,0]*10.0
    logits = u[:,1:]
    # ac = tf.py_func(cat_sample, [p], [tf.float32])[0]
    ac = tf.multinomial(logits, 1)[:,0]
    alllogp = tf.nn.log_softmax(logits)
    logp = U.fancy_slice_2d(alllogp, tf.range(batch_size), ac)
    ent = - U.sum(alllogp * tf.exp(alllogp), axis=1)
    return ac, logp, vpred, ent

# TODO fix target values
alg = OnlineVPG(ve, policy, horizon=50)
alg.train(1000)