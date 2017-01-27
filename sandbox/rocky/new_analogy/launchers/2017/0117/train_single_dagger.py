import pickle

import gym
from gym.spaces import prng
import matplotlib.pyplot as plt

import gpr_package.bin.tower_fetch_policy as tower
from gpr.envs.fetch.sim_fetch import Experiment as SimFetch
import time
import numpy as np
import sandbox.rocky.tf.core.layers as L
from gpr.utils.space_utils import flatten
from rllab.misc.ext import AttrDict
from rllab.sampler.utils import rollout
from sandbox.rocky.new_analogy.pymj import MjParallelLite
from sandbox.rocky.s3.resource_manager import resource_manager

T = 500

task_id = tower.get_task_from_text("ab")


def get_demonstration():
    expr = SimFetch(nboxes=2, horizon=T, mocap=True, obs_type="full_state")
    env = expr.make(task_id)

    # obtain demonstration
    policy = tower.FetchPolicy(task_id)
    env.seed(0)
    prng.seed(0)
    xinit = env.world.sample_xinit()
    xs = []
    acts = []
    ob = env.reset_to(xinit)
    obs = []
    rewards = []
    for t in range(T):
        a = policy.get_action(ob)
        xs.append(env.x)
        acts.append(a)
        obs.append(env.observation_space.flatten(ob))
        ob, reward, _, _ = env.step(a)
        rewards.append(reward)
    xs = np.asarray(xs)
    acts = np.asarray(acts)
    obs = np.asarray(obs)

    # collect site xpos
    # the shaping cost will be
    dimq = env.world.dimq
    qpos, qvel = xs[:, :dimq], xs[:, dimq:]

    site_xpos = np.zeros((T, env.world.dimxpos))

    def lambda_over_sense(sense, i):
        site_xpos[i] = sense["site_xpos"]

    env.world.model.forward(qpos, qvel, lambda_over_sense)

    return dict(
        observations=obs,
        actions=acts,
        rewards=np.asarray(rewards),
        env_infos=dict(
            site_xpos=site_xpos,
            x=xs,
        )
    )


class WrappedEnv(object):
    def __init__(self, env, xinit, acts):
        self.env = env
        self.xinit = xinit
        acts = acts[:, [0, 1, 2, 6, 7]]
        self.action_bounds = (acts.min(axis=0), acts.max(axis=0))

    def normalize_actions(self, actions):
        actions = np.asarray(actions)[:, [0, 1, 2, 6, 7]]
        lb, ub = self.action_bounds
        # scale to [0, 1]
        actions = (actions - lb) / (ub - lb)
        # import ipdb; ipdb.set_trace()
        # transform to [-1, 1]
        actions = actions * 2 - 1
        return actions

    def unnormalize_actions(self, actions):
        actions = np.asarray(actions)
        lb, ub = self.action_bounds
        actions = (actions + 1) / 2 * (ub - lb) + lb
        N = len(actions)
        return np.asarray([
            actions[:, 0],
            actions[:, 1],
            actions[:, 2],
            np.zeros(N),
            np.ones(N) * 1.57,
            np.zeros(N),
            actions[:, 3],
            actions[:, 4],
        ]).T

    def reset(self):
        return self.env.reset_to(self.xinit)

    def reset_to(self, xinit):
        return self.env.reset_to(xinit)

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        space = gym.spaces.Box(-1e3, 1e3, (5,))
        space.flatten_dim = 5
        space.flatten = flatten
        return space

    def step(self, action):
        action = self.unnormalize_actions([action])[0]
        return self.env.step(action)


def train_policy():
    import tensorflow as tf
    expr = SimFetch(nboxes=2, horizon=T, mocap=True, obs_type="full_state")
    env = expr.make(task_id)

    # import ipdb; ipdb.set_trace()
    print("Generating demonstrations")
    xs, acts = get_demonstration()
    obs = []
    print("Converting to obs")
    for x in xs:
        obs.append(env.observation_space.flatten(env.world.observe(x)[0]))
    obs = np.asarray(obs)

    # normalize actions
    env = WrappedEnv(env, xs[0], acts)
    acts = env.normalize_actions(acts)
    obs_dim = env.observation_space.flatten_dim
    action_dim = env.action_space.flatten_dim

    # try separate networks for each action dimension

    l_in = L.InputLayer(shape=(None, obs_dim))
    # l_outs = []

    # for action_idx in range(action_dim):
    #     # n_mixtures = 10
    l_h1 = L.DenseLayer(l_in, num_units=256, nonlinearity=tf.nn.tanh)
    l_h2 = L.DenseLayer(l_h1, num_units=256, nonlinearity=tf.nn.tanh)
    l_out = L.DenseLayer(l_h2, num_units=action_dim, nonlinearity=tf.nn.tanh)
    # l_softmax = L.DenseLayer(l_h2, num_units=n_mixtures, nonlinearity=tf.nn.softmax)
    # l_mixturevals = L.DenseLayer(l_h2, num_units=n_mixtures, nonlinearity=tf.nn.tanh)
    # l_out = L.OpLayer(
    #     incoming=l_softmax,
    #     extras=[l_mixturevals],
    #     op=lambda x, y: tf.reduce_sum(x * y, -1, keep_dims=True),
    #     shape_op=lambda x, y: x[:-1] + (1,)
    # )
    #     l_outs.append(l_out)
    #
    # l_out = L.concat(l_outs)

    params = L.get_all_params(l_out)

    # train using squared loss

    obs_var = l_in.input_var
    action_var = L.get_output(l_out)
    ref_action_var = tf.placeholder(dtype=tf.float32, shape=(None, action_dim), name="ref_action")

    loss_var = tf.reduce_mean(tf.square(action_var - ref_action_var))

    train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss_var, var_list=params)

    batch_size = 128

    with tf.Session() as sess:

        sess.run(tf.initialize_all_variables())

        running_loss = 0
        itr = 0

        def get_action(obs):
            obs = env.observation_space.flatten(obs)
            action = sess.run(action_var, feed_dict={obs_var: [obs]})[0]
            return action, dict()

        while True:
            ids = np.random.choice(T, size=batch_size, replace=False)
            batch_obs = obs[ids]
            batch_acts = acts[ids]
            _, loss = sess.run([train_op, loss_var], feed_dict={obs_var: batch_obs, ref_action_var: batch_acts})

            running_loss = running_loss * 0.99 + loss * 0.01
            itr += 1

            if itr % 100 == 0:
                print("Running loss: ", running_loss)

                path = rollout(env, AttrDict(get_action=get_action, reset=lambda *_: 0), max_path_length=T)
                print(path["rewards"][-1])


if __name__ == "__main__":
    path = get_demonstration()
    resource_manager.register_data("fetch_single_traj", pickle.dumps(path))
    # site_xpos = path["env_infos"]["site_xpos"]
    # # obs_dim = obs.shape[-1]
    # for idx in range(site_xpos.shape[-1]):
    #     plt.subplot(3, 3, idx + 1)
    #     plt.plot(site_xpos[:, idx])  # all_actions[:, idx], 100)
    # plt.show()
    # import ipdb; ipdb.set_trace()
    # train_policy()
