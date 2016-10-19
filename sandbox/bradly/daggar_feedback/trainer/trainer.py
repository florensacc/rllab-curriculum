from collections import OrderedDict

from rllab.core.serializable import Serializable
from rllab.misc import logger
import numpy as np
import tensorflow as tf
import pyprind
from rllab.sampler.utils import rollout
from sandbox.rocky.analogy.policies.apply_demo_policy import ApplyDemoPolicy
from sandbox.rocky.analogy.dataset import SupervisedDataset
from sandbox.bradly.analogy.policy.non_broken_normalizing_policy import NormalizingPolicy
from sandbox.rocky.analogy.utils import unwrap
from rllab.sampler.stateful_pool import singleton_pool
import itertools
import random
import contextlib

from sandbox.rocky.tf.envs.vec_env_executor import VecEnvExecutor
from sandbox.rocky.tf.misc import tensor_utils

from sandbox.rocky.analogy.envs.conopt_particle_env import ConoptParticleEnv

# A simple example hopefully able to train a feed-forward network


class Trainer(Serializable):
    def __init__(
            self,
            policy,
            env_cls,
            demo_collector,
            n_train_trajs=50,
            n_test_trajs=20,
            horizon=50,
            batch_size=10,
            n_epochs=100,
            n_passes_per_epoch=1,
            n_eval_trajs=10,
            learning_rate=1e-3,
            plot=False,
    ):
        Serializable.quick_init(self, locals())
        self.env_cls = env_cls
        self.demo_collector = demo_collector
        self.n_train_trajs = n_train_trajs
        self.n_test_trajs = n_test_trajs
        self.horizon = horizon
        self.policy = policy
        self.plot = plot
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_passes_per_epoch = n_passes_per_epoch
        self.n_eval_trajs = n_eval_trajs
        self.learning_rate = learning_rate

    def train(self, n_samps=200):
        train_op, train_loss, expert_actions_tf, expert_obs_tf = self.init_tf_vars(self.env_cls(), self.policy)
        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)
        self.policy.sess = sess

        #expert_samples = self.collect_supervision_data(self.demo_collector, self.env_cls, self.horizon,
        #                                               n_samps, self.policy)
        #EXPERT_OBS_NP, EXPERT_ACTIONS_NP = self.convert_expert_to_np(expert_samples)

        epoch_losses = []
        for epoch_step in range(0, self.n_epochs):
            losses = []
            expert_samples = self.collect_supervision_data(self.demo_collector, self.env_cls, self.horizon,
                                                       n_samps, self.policy)
            EXPERT_OBS_NP, EXPERT_ACTIONS_NP = self.convert_expert_to_np(expert_samples)
            for batch_step in range(0, EXPERT_OBS_NP.shape[0], self.batch_size):
                local_obs = EXPERT_OBS_NP[batch_step: batch_step+self.batch_size]
                local_actions = EXPERT_ACTIONS_NP[batch_step: batch_step+self.batch_size]
                loss = sess.run([train_op, train_loss], feed_dict={expert_obs_tf: local_obs,
                                                                   expert_actions_tf: local_actions})[1]
                losses.append(loss)
            #print(losses)
            this_epoch_loss = np.mean(np.asarray(losses))
            epoch_losses.append(this_epoch_loss)
            print(this_epoch_loss)

    def init_tf_vars(self, env, policy):
        # everything has 1 extra dims because of the batch size

        expert_obs = policy.obs_var
        policy_actions = policy.action_var

        expert_actions = env.action_space.new_tensor_variable(name="expert_actions", extra_dims=1)
        train_loss_var = tf.reduce_mean(tf.square(policy_actions - expert_actions))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        params = policy.get_params(trainable=True)
        grads_and_vars = optimizer.compute_gradients(train_loss_var, var_list=params)
        train_op = optimizer.apply_gradients(grads_and_vars)
        return train_op, train_loss_var, expert_actions, expert_obs

    @staticmethod
    # get supervised data from the expert.
    def collect_supervision_data(demo_collector, env_cls, horizon, n_samps, novice):
        paths = []
        progbar = pyprind.ProgBar(n_samps)
        for iter_step in range(0, n_samps):
            demo_env = env_cls()
            demo_path = demo_collector.collect_demo(env=demo_env, novice=novice, horizon=horizon)
            paths.append(demo_path)
            progbar.update()
        if progbar.active:
            progbar.stop()
        return paths

    def convert_expert_to_np(self, expert_data):
        #obs_dim = expert_data[0]['observations'][0].flatten().shape[0]
        #act_dim = expert_data[0]['actions'][0].flatten().shape[0]
        obs_dim = self.policy.obs_dim #tf.shape(expert_obs)[0]
        act_dim = self.policy.action_dim #tf.shape(expert_actions)[0]
        data_size = len(expert_data)*len(expert_data[0]['observations'])
        exp_obs = np.zeros(shape=(data_size, obs_dim))
        exp_actions = np.zeros(shape=(data_size, act_dim))

        iter_step = 0
        for one_path in expert_data:
            for one_obs_step, one_act_step in zip(one_path['observations'], one_path['actions']):
                exp_obs[iter_step, :] = one_obs_step
                exp_actions[iter_step, :] = one_act_step
                iter_step += 1
        return exp_obs, exp_actions



    @staticmethod
    # rollout the policy.
    def get_on_policy_rollouts(policy, env, max_path_length, n_rollouts):
        paths = []
        progbar = pyprind.ProgBar(n_rollouts)
        for iter_step in range(0, n_rollouts):
            paths.append(rollout(env=env, agent=policy, max_path_length=max_path_length))
            progbar.update()
        if progbar.active:
            progbar.stop()
        return paths