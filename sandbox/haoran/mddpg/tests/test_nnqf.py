import unittest

import numpy as np
import tensorflow as tf

import qfunctions.nn_qfunction


class TestFeedForwardCritic(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        self.sess = tf.Session()
        self.sess.__enter__()

    def tearDown(self):
        self.sess.__exit__(None, None, None)

    def test_copy(self):
        action_dim = 5
        obs_dim = 7
        critic1 = qfunctions.nn_qfunction.FeedForwardCritic("1", obs_dim, action_dim)
        critic2 = qfunctions.nn_qfunction.FeedForwardCritic("2", obs_dim, action_dim)
        critic1.sess = self.sess
        critic2.sess = self.sess

        a = np.random.rand(1, action_dim)
        o = np.random.rand(1, obs_dim)

        feed_1 = {
            critic1.actions_placeholder: a,
            critic1.observations_placeholder: o,
        }
        feed_2 = {
            critic2.actions_placeholder: a,
            critic2.observations_placeholder: o,
        }

        self.sess.run(tf.initialize_all_variables())

        out1 = self.sess.run(critic1.output, feed_1)
        out2 = self.sess.run(critic2.output, feed_2)
        self.assertFalse((out1 == out2).all())

        critic2.set_param_values(critic1.get_param_values())
        out1 = self.sess.run(critic1.output, feed_1)
        out2 = self.sess.run(critic2.output, feed_2)
        self.assertTrue((out1 == out2).all())

    def test_output_len(self):
        action_dim = 5
        obs_dim = 7
        critic = qfunctions.nn_qfunction.FeedForwardCritic("1", obs_dim, action_dim)
        critic.sess = self.sess

        a = np.random.rand(1, action_dim)
        o = np.random.rand(1, obs_dim)
        feed = {
            critic.actions_placeholder: a,
            critic.observations_placeholder: o,
        }

        self.sess.run(tf.initialize_all_variables())

        out = self.sess.run(critic.output, feed)
        self.assertEqual(1, out.size)