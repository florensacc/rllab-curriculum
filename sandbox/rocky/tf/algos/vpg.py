from __future__ import print_function
from __future__ import absolute_import
from rllab.algos.batch_polopt import BatchPolopt
import tensorflow as tf
from rllab.misc import logger
from rllab.misc import ext


class VPG(BatchPolopt):
    def __init__(self, learning_rate, *args, **kwargs):
        BatchPolopt.__init__(self, *args, **kwargs)
        self.learning_rate = learning_rate
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.input_list = None
        self.kl_input_list = None
        self.train_op = None
        self.mean_kl_var = None
        self.max_kl_var = None
        self.surr_obj_var = None

    def init_opt(self):
        assert not self.policy.recurrent

        obs_var = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1,
        )
        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1,
        )
        advantage_var = tf.placeholder(tf.float32, shape=[None], name='advantage')
        dist = self.policy.distribution
        old_dist_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] + list(shape), name='old_%s' % k)
            for k, shape in dist.dist_info_specs
            }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

        state_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] + list(shape), name=k)
            for k, shape in self.policy.state_info_specs
            }
        state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]

        dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)
        logli = dist.log_likelihood_sym(action_var, dist_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)

        # formulate as a minimization problem
        # The gradient of the surrogate objective is the policy gradient
        surr_obj = - tf.reduce_mean(logli * advantage_var)
        mean_kl = tf.reduce_mean(kl)
        max_kl = tf.reduce_max(kl)

        self.input_list = [obs_var, action_var, advantage_var] + state_info_vars_list
        self.kl_input_list = self.input_list + old_dist_info_vars_list
        self.train_op = self.optimizer.minimize(surr_obj, var_list=self.policy.get_params(trainable=True))
        self.mean_kl_var = mean_kl
        self.max_kl_var = max_kl
        self.surr_obj_var = surr_obj
        tf.get_default_session().run(tf.initialize_all_variables())

    def optimize_policy(self, itr, samples_data):
        logger.log("optimizing policy")
        sess = tf.get_default_session()
        inputs = ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        )
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        inputs += tuple(state_info_list)
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        feed_dict = dict(zip(self.input_list, inputs))
        loss_before = sess.run(self.surr_obj_var, feed_dict=feed_dict)
        sess.run(self.train_op, feed_dict=feed_dict)
        loss_after = sess.run(self.surr_obj_var, feed_dict=feed_dict)
        logger.record_tabular("LossBefore", loss_before)
        logger.record_tabular("LossAfter", loss_after)

        kl_feed_dict = dict(zip(self.kl_input_list, list(inputs) + dist_info_list))

        mean_kl, max_kl = sess.run([self.mean_kl_var, self.max_kl_var], feed_dict=kl_feed_dict)
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('MaxKL', max_kl)

    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )
