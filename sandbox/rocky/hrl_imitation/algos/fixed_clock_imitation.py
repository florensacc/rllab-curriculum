from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import tensorflow as tf

from rllab.algos.base import RLAlgorithm
from rllab.misc import logger
from sandbox.rocky.tf.core.parameterized import JointParameterized
from sandbox.rocky.tf.misc import tensor_utils


class FixedClockImitation(RLAlgorithm):
    def __init__(
            self,
            env_expert,
            policy_module,
            recog,
            subgoal_dim=4,
            subgoal_interval=3,
            bottleneck_dim=10,
            batch_size=128,
            learning_rate=1e-3,
            discount=0.99,
            max_path_length=100,
            bottleneck_coeff=0.,
            n_epochs=100,
            n_sweep_per_epoch=10,
    ):
        self.env_expert = env_expert
        self.policy_module = policy_module
        self.recog = recog
        self.subgoal_dim = subgoal_dim
        self.subgoal_interval = subgoal_interval
        self.bottleneck_dim = bottleneck_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.discount = discount
        self.max_path_length = max_path_length
        self.bottleneck_coeff = bottleneck_coeff
        self.n_epochs = n_epochs
        self.n_sweep_per_epoch = n_sweep_per_epoch

        self.high_policy = None
        self.alt_high_policy = None
        self.low_policy = None

        self.f_train = None

    def build_policy(self):
        env_spec = self.env_expert.env_spec
        self.high_policy = self.policy_module.new_high_policy(
            env_spec=env_spec,
            subgoal_dim=self.subgoal_dim
        )
        self.alt_high_policy = self.policy_module.new_alt_high_policy(
            env_spec=env_spec,
            subgoal_dim=self.subgoal_dim
        )
        self.low_policy = self.policy_module.new_low_policy(
            env_spec=env_spec,
            subgoal_dim=self.subgoal_dim,
            bottleneck_dim=self.bottleneck_dim
        )

    def init_opt(self):
        logger.log("setting up training")

        env_spec = self.env_expert.env_spec
        obs_dim = env_spec.observation_space.flat_dim
        action_dim = env_spec.action_space.flat_dim

        # There's the option for batch optimization vs. minibatch
        # Since we need to sample from the approx. posterior anyways, we'll go for minibatch
        obs_var = env_spec.observation_space.new_tensor_variable(
            name="obs",
            extra_dims=2,
        )
        action_var = env_spec.action_space.new_tensor_variable(
            name="action",
            extra_dims=2,
        )
        bottleneck_epsilon_var = tf.placeholder(
            dtype=tf.float32,
            shape=(None, self.bottleneck_dim),
            name="bottleneck_epsilon"
        )
        # Sample h~q(h|s, a)
        # Should return the same dimension
        recog_subgoal_dist = self.recog.dist_info_sym(obs_var, action_var)
        recog_subgoal = self.recog.distribution.sample_sym(recog_subgoal_dist)
        flat_obs_var = tf.reshape(obs_var, (-1, obs_dim))
        flat_action_var = tf.reshape(action_var, (-1, action_dim))

        high_obs_var = obs_var[:, 0, :]
        policy_subgoal_dist = self.high_policy.dist_info_sym(high_obs_var, dict())

        # tile the subgoals to match the dimension of obs / actions
        tiled_recog_subgoal = tf.tile(
            tf.expand_dims(recog_subgoal, 1),
            [1, self.subgoal_interval, 1]
        )

        flat_recog_subgoal = tf.reshape(tiled_recog_subgoal, (-1, self.subgoal_dim))

        low_obs = tf.concat(1, [flat_obs_var, flat_recog_subgoal])
        action_dist_info = self.low_policy.dist_info_sym(low_obs, dict(bottleneck_epsilon=bottleneck_epsilon_var))
        flat_action_logli = self.low_policy.distribution.log_likelihood_sym(flat_action_var, action_dist_info)

        action_logli = tf.reshape(flat_action_logli, (-1, self.subgoal_interval))
        sum_action_logli = tf.reduce_sum(action_logli, -1)

        subgoal_kl = self.high_policy.distribution.kl_sym(recog_subgoal_dist, policy_subgoal_dist)
        subgoal_logli = self.recog.distribution.log_likelihood_sym(recog_subgoal, recog_subgoal_dist)

        bottleneck_dist_info = self.low_policy.bottleneck_dist_info_sym(low_obs)
        prior_bottleneck_dist_info = dict(
            mean=tf.zeros_like(bottleneck_dist_info["mean"]),
            log_std=tf.zeros_like(bottleneck_dist_info["log_std"])
        )
        bottleneck_kl = self.low_policy.bottleneck_dist.kl_sym(bottleneck_dist_info, prior_bottleneck_dist_info)

        vlb = tf.reduce_mean(- sum_action_logli + subgoal_kl)
        avg_bottleneck_kl = tf.reduce_mean(bottleneck_kl)

        loss = vlb + self.bottleneck_coeff * avg_bottleneck_kl
        surr_loss = loss - tf.reduce_mean(tf.stop_gradient(sum_action_logli) * subgoal_logli)

        joint_target = JointParameterized([self.high_policy, self.low_policy, self.recog])

        params = joint_target.get_params(trainable=True)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_op = optimizer.minimize(surr_loss, var_list=params)

        self.f_train = tensor_utils.compile_function(
            inputs=[obs_var, action_var, bottleneck_epsilon_var],
            outputs=[train_op, loss, vlb, avg_bottleneck_kl],
        )

    def train(self):
        self.build_policy()
        self.init_opt()
        dataset = self.env_expert.build_dataset(self.batch_size)

        logger.log("start training")
        with tf.Session() as sess:
            logger.log("initializing variables")
            sess.run(tf.initialize_all_variables())
            logger.log("initialized")

            for epoch_id in xrange(self.n_epochs):

                losses = []
                vlbs = []
                bottleneck_kls = []

                logger.log("Start epoch %d..." % epoch_id)

                for _ in xrange(self.n_sweep_per_epoch):

                    for batch_obs, batch_actions in dataset.iterate():
                        # Sample minibatch and train
                        N = batch_obs.shape[0] * batch_obs.shape[1]
                        epsilons = np.random.normal(size=(N, self.bottleneck_dim))
                        _, loss_val, vlb_val, bottleneck_kl_val = self.f_train(batch_obs, batch_actions, epsilons)
                        losses.append(loss_val)
                        vlbs.append(vlb_val)
                        bottleneck_kls.append(bottleneck_kl_val)

                logger.log("Evaluating...")

                logger.record_tabular("Epoch", epoch_id)
                logger.record_tabular("AverageLoss", np.mean(losses))
                logger.record_tabular("AverageNegVlb", np.mean(vlbs))
                logger.record_tabular("AverageBottleneckKL", np.mean(bottleneck_kls))
                self.env_expert.log_diagnostics(self)
                logger.dump_tabular()
