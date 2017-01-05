import unittest

import numpy as np
import tensorflow as tf

import policies.nn_policy
import qfunctions.nn_qfunction
from algos.ddpg import DDPG
from misc.testing_utils import are_np_arrays_equal, are_np_array_lists_equal
from misc.tf_test_case import TFTestCase
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.exploration_strategies.ou_strategy import OUStrategy
from sandbox.rocky.tf.envs.base import TfEnv


class TestDDPG(TFTestCase):

    def setUp(self):
        super().setUp()
        self.env = TfEnv(CartpoleEnv())
        self.es = OUStrategy(env_spec=self.env.spec)

    def test_target_params_copied(self):
        algo = DDPG(
            self.env,
            self.es,
            policies.nn_policy.SumPolicy('policies', 4, 1),
            qfunctions.nn_qfunction.SumCritic('qf', 4, 1),
            n_epochs=0,
        )
        target_critic = algo.target_qf
        target_actor = algo.target_policy
        critic = algo.qf
        critic_copy = algo.critic_with_action_input
        actor = algo.policy

        # Make sure they're different to start
        random_values = [np.random.rand(*values.shape)
                         for values in critic.get_param_values()]
        critic.set_param_values(random_values)
        random_values = [np.random.rand(*values.shape)
                         for values in actor.get_param_values()]
        actor.set_param_values(random_values)

        self.assertParamsNotEqual(target_critic, critic)
        self.assertParamsNotEqual(target_actor, actor)
        self.assertParamsEqual(critic_copy, critic)

        algo.train()
        self.assertParamsEqual(target_critic, critic)
        self.assertParamsEqual(target_actor, actor)
        self.assertParamsEqual(critic_copy, critic)

    def test_target_params_update(self):
        tau = 0.2
        algo = DDPG(
            self.env,
            self.es,
            policies.nn_policy.SumPolicy('policies', 4, 1),
            qfunctions.nn_qfunction.SumCritic('qf', 4, 1),
            n_epochs=0,
            soft_target_tau=tau,
        )
        target_critic = algo.target_qf
        target_actor = algo.target_policy
        critic = algo.qf
        actor = algo.policy

        algo.train()

        orig_tc_vals = target_critic.get_param_values()
        orig_ta_vals = target_actor.get_param_values()
        orig_c_vals = critic.get_param_values()
        orig_a_vals = actor.get_param_values()
        algo.sess.run(algo.update_target_actor_op)
        algo.sess.run(algo.update_target_critic_op)
        new_tc_vals = target_critic.get_param_values()
        new_ta_vals = target_actor.get_param_values()

        for orig_tc_val, orig_c_val, new_tc_val in zip(
                orig_tc_vals, orig_c_vals, new_tc_vals):
            self.assertTrue(
                (new_tc_val == tau * orig_c_val + (1-tau) * orig_tc_val).all())

        for orig_ta_val, orig_a_val, new_ta_val in zip(
                orig_ta_vals, orig_a_vals, new_ta_vals):
            self.assertTrue(
                (new_ta_val == tau * orig_a_val + (1-tau) * orig_ta_val).all())

    def test_target_params_hard_update(self):
        tau = 1
        algo = DDPG(
            self.env,
            self.es,
            policies.nn_policy.SumPolicy('policies', 4, 1),
            qfunctions.nn_qfunction.SumCritic('qf', 4, 1),
            n_epochs=0,
            soft_target_tau=tau,
        )
        target_critic = algo.target_qf
        target_actor = algo.target_policy
        critic = algo.qf
        actor = algo.policy

        random_values = [np.random.rand(*values.shape)
                         for values in critic.get_param_values()]
        critic.set_param_values(random_values)
        random_values = [np.random.rand(*values.shape)
                         for values in actor.get_param_values()]
        actor.set_param_values(random_values)
        self.assertParamsNotEqual(target_critic, critic)
        self.assertParamsNotEqual(target_actor, actor)
        algo.sess.run(algo.update_target_actor_op)
        algo.sess.run(algo.update_target_critic_op)
        self.assertParamsEqual(target_critic, critic)
        self.assertParamsEqual(target_actor, actor)

    def test_target_params_no_update(self):
        tau = 0
        algo = DDPG(
            self.env,
            self.es,
            policies.nn_policy.SumPolicy('policies', 4, 1),
            qfunctions.nn_qfunction.SumCritic('qf', 4, 1),
            n_epochs=0,
            soft_target_tau=tau,
        )
        target_critic = algo.target_qf
        target_actor = algo.target_policy
        critic = algo.qf
        actor = algo.policy

        random_values = [np.random.rand(*values.shape)
                         for values in critic.get_param_values()]
        critic.set_param_values(random_values)
        random_values = [np.random.rand(*values.shape)
                         for values in actor.get_param_values()]
        actor.set_param_values(random_values)
        old_target_critic_values = target_critic.get_param_values()
        old_target_actor_values = target_actor.get_param_values()
        self.assertParamsNotEqual(target_critic, critic)
        self.assertParamsNotEqual(target_actor, actor)
        algo.sess.run(algo.update_target_actor_op)
        algo.sess.run(algo.update_target_critic_op)
        self.assertTrue(are_np_array_lists_equal(
            old_target_critic_values,
            target_critic.get_param_values()
        ))
        self.assertTrue(are_np_array_lists_equal(
            old_target_actor_values,
            target_actor.get_param_values()
        ))
        self.assertParamsNotEqual(target_critic, critic)
        self.assertParamsNotEqual(target_actor, actor)

    def test_sum_critic(self):
        algo = DDPG(
            self.env,
            self.es,
            policies.nn_policy.SumPolicy('policies', 4, 1),
            qfunctions.nn_qfunction.SumCritic('qf', 4, 1),
            n_epochs=0,
            epoch_length=0,
            eval_samples=0,  # Ignore eval. Just do this to remove warnings.
        )
        obs = np.array([[1., 1., 1., 1.]])
        actions = np.array([[-0.5]])
        for critic in [algo.qf, algo.target_qf]:
            feed_dict = {
                critic.actions_placeholder: actions,
                critic.observations_placeholder: obs,
            }
            self.assertEqual(
                np.sum(obs) + actions,
                algo.sess.run(critic.output, feed_dict=feed_dict)
            )

    def test_sum_policy(self):
        algo = DDPG(
            self.env,
            self.es,
            policies.nn_policy.SumPolicy('policies', 4, 1),
            qfunctions.nn_qfunction.SumCritic('qf', 4, 1),
            n_epochs=0,
            epoch_length=0,
            eval_samples=0,  # Ignore eval. Just do this to remove warnings.
        )
        obs = np.array([[1., 1., 1., 1.]])
        for actor in [algo.policy, algo.target_policy]:
            feed_dict = {
                actor.observations_placeholder: obs,
            }
            self.assertEqual(
                np.sum(obs),
                algo.sess.run(actor.output, feed_dict=feed_dict)
            )

    def test_critic_targets(self):
        discount = 0.5
        algo = DDPG(
            self.env,
            self.es,
            policies.nn_policy.SumPolicy('policies', 4, 1),
            qfunctions.nn_qfunction.SumCritic('qf', 4, 1),
            n_epochs=0,
            epoch_length=0,
            eval_samples=0,  # Ignore eval. Just do this to remove warnings.
            discount=discount,
        )

        rewards = np.array([3., 4.])
        terminals = np.array([0., 0.])
        obs = np.array([[1., 1., 1., 1.], [1., 1., 1., 1.]])
        actions = np.array([[-0.5], [-0.5]])
        next_obs = np.array([[1., 1., 1., 1.], [1., 1., 1., 1.]])

        # target = reward + discount * target_critic(next_obs,
        #                                            target_actor(next_obs))
        # target1 = 3 + 0.5 * Q([1,1,1,1], u([1,1,1,1]))
        #         = 3 + 0.5 * Q([1,1,1,1], 4)
        #         = 3 + 0.5 * 8
        #         = 7
        # target2 = 8

        feed_dict = algo._critic_feed_dict(rewards,
                                           terminals,
                                           obs,
                                           actions,
                                           next_obs)
        self.assertNpEqual(
            np.array([[7.], [8.]]),
            algo.sess.run(algo.ys, feed_dict=feed_dict)
        )

    def test_critic_targets2(self):
        discount = 0.5
        algo = DDPG(
            self.env,
            self.es,
            policies.nn_policy.SumPolicy('policies', 4, 1),
            qfunctions.nn_qfunction.SumCritic('qf', 4, 1),
            n_epochs=0,
            epoch_length=0,
            eval_samples=0,  # Ignore eval. Just do this to remove warnings.
            discount=discount,
        )

        rewards = np.array([3.5])
        terminals = np.array([0.])
        obs = np.array([[1., 1., 1., 1.]])
        actions = np.array([[2.]])
        next_obs = np.array([[2., 2., 2., 2.]])

        # target = reward + discount * target_critic(next_obs,
        #                                            target_actor(next_obs))
        # target = 3.5 + 0.5 * Q([2,2,2,2], u([2,2,2,2]))
        #        = 3.5 + 0.5 * Q([2,2,2,2], 8)
        #        = 3.5 + 0.5 * 16
        #        = 11.5
        feed_dict = algo._critic_feed_dict(rewards,
                                           terminals,
                                           obs,
                                           actions,
                                           next_obs)
        self.assertNpEqual(
            np.array([[11.5]]),
            algo.sess.run(algo.ys, feed_dict=feed_dict)
        )

    def test_critic_loss(self):
        discount = 0.5
        algo = DDPG(
            self.env,
            self.es,
            policies.nn_policy.SumPolicy('policies', 4, 1),
            qfunctions.nn_qfunction.SumCritic('qf', 4, 1),
            n_epochs=0,
            epoch_length=0,
            eval_samples=0,  # Ignore eval. Just do this to remove warnings.
            discount=discount,
        )

        rewards = np.array([3.])
        terminals = np.array([0.])
        obs = np.array([[1., 1., 1., 1.]])
        actions = np.array([[-0.5]])
        next_obs = np.array([[1., 1., 1., 1.]])

        # target = reward + discount * target_critic(next_obs,
        #                                            target_actor(next_obs))
        # target = 3 + 0.5 * Q([1,1,1,1], u([1,1,1,1]))
        #        = 3 + 0.5 * Q([1,1,1,1], 4)
        #        = 3 + 0.5 * 8
        #        = 7
        #
        # loss = (target - critic(obs, action))^2
        # loss = (target - critic([1,1,1,1], -0.5))^2
        #      = (target - 3.5)^2
        #      = (7 - 3.5)^2
        #      = (3.5)^2
        #      = 12.25
        feed_dict = algo._critic_feed_dict(rewards,
                                           terminals,
                                           obs,
                                           actions,
                                           next_obs)
        actual = algo.sess.run(algo.critic_loss, feed_dict=feed_dict)
        self.assertEqual(
            12.25,
            actual
        )
        self.assertEqual(
            np.float32,
            type(actual)
        )

    def test_critic_loss2(self):
        discount = 0.5
        algo = DDPG(
            self.env,
            self.es,
            policies.nn_policy.SumPolicy('policies', 4, 1),
            qfunctions.nn_qfunction.SumCritic('qf', 4, 1),
            n_epochs=0,
            epoch_length=0,
            eval_samples=0,  # Ignore eval. Just do this to remove warnings.
            discount=discount,
        )

        rewards = np.array([3.5])
        terminals = np.array([0.])
        obs = np.array([[1., 1., 1., 1.]])
        actions = np.array([[2.]])
        next_obs = np.array([[2., 2., 2., 2.]])

        # target = reward + discount * target_critic(next_obs,
        #                                            target_actor(next_obs))
        # target = 3.5 + 0.5 * Q([2,2,2,2], u([2,2,2,2]))
        #        = 3.5 + 0.5 * Q([2,2,2,2], 8)
        #        = 3.5 + 0.5 * 16
        #        = 11.5
        #
        # loss = (target - critic(obs, action))^2
        #      = (target - critic([1,1,1,1], 2))^2
        #      = (target - 6)^2
        #      = (11.5 - 6)^2
        #      = (5.5)^2
        #      = 30.25
        feed_dict = algo._critic_feed_dict(rewards,
                                           terminals,
                                           obs,
                                           actions,
                                           next_obs)
        actual = algo.sess.run(algo.critic_loss, feed_dict=feed_dict)
        self.assertEqual(
            30.25,
            actual
        )
        self.assertEqual(
            np.float32,
            type(actual)
        )

    def test_critic_gradient(self):
        discount = 0.5
        algo = DDPG(
            self.env,
            self.es,
            policies.nn_policy.SumPolicy('policies', 4, 1),
            qfunctions.nn_qfunction.SumCritic('qf', 4, 1),
            n_epochs=0,
            epoch_length=0,
            eval_samples=0,  # Ignore eval. Just do this to remove warnings.
            discount=discount,
        )

        rewards = np.array([3.5])
        terminals = np.array([0.])
        obs = np.array([[1., 1., 1., 1.]])
        actions = np.array([[1.]])
        next_obs = np.array([[2., 2., 2., 2.]])

        # target = reward + discount * target_critic(next_obs,
        #                                            target_actor(next_obs))
        # target = 3.5 + 0.5 * Q([2,2,2,2], u([2,2,2,2]))
        #        = 3.5 + 0.5 * Q([2,2,2,2], 8)
        #        = 3.5 + 0.5 * 16
        #        = 11.5
        #
        # dloss/dtheta = - 2 ( y - critic(obs, action)) *
        #                   d/dtheta (critic(obs, action))
        # dloss/dtheta = - 2 ( y - critic([1,1,1,1], 1)) *
        #                   d/dtheta (critic(obs, action))
        # dloss/dtheta = - 2 ( 11.5 - 5) *
        #                   d/dtheta (critic(obs, action))
        # dloss/dtheta = - 13 * d/dtheta (critic(obs, action))
        feed_dict = algo._critic_feed_dict(rewards,
                                           terminals,
                                           obs,
                                           actions,
                                           next_obs)
        grads = tf.gradients(algo.critic_loss, algo.qf.get_params_internal())
        # critic_grads = algo.sess.run(
        #         tf.gradients(algo.critic.output, algo.critic.get_vars()))
        expected = [-13. * np.ones_like(v)
                    for v in algo.qf.get_param_values()]
        actual = algo.sess.run(grads, feed_dict=feed_dict)
        actual_flat = np.vstack(actual).flatten()
        self.assertTrue(
            are_np_array_lists_equal(expected, actual_flat),
            "Numpy arrays not equal")

    def test_actor_surrogate_loss(self):
        discount = 0.5
        algo = DDPG(
            self.env,
            self.es,
            policies.nn_policy.SumPolicy('policies', 4, 1),
            qfunctions.nn_qfunction.SumCritic('qf', 4, 1),
            n_epochs=0,
            epoch_length=0,
            eval_samples=0,  # Ignore eval. Just do this to remove warnings.
            discount=discount,
        )

        obs = np.array([[1., 1., 1., 1.],
                        [10., 10., 10., -10.]])

        # loss = -1/N sum_i Q(s_i, u(s_i))
        #      = -1/2 * {(Q([1,1,1,1], u([1,1,1,1]))
        #                + Q([10,10,10,-10], u([10,10,10,-10]))}
        #      = -1/2 * {Q([1,1,1,1], 4)) + Q([10,10,10,-10], 20))}
        #      = -1/2 * (8 + 40)
        #      = -24
        feed_dict = algo._actor_feed_dict(obs)
        actual = algo.sess.run(algo.actor_surrogate_loss, feed_dict=feed_dict)
        self.assertEqual(actual, -24.)
        self.assertEqual(
            np.float32,
            type(actual)
        )

    def test_actor_surrogate_loss2(self):
        discount = 0.5
        algo = DDPG(
            self.env,
            self.es,
            policies.nn_policy.SumPolicy('policies', 4, 1),
            qfunctions.nn_qfunction.SumCritic('qf', 4, 1),
            n_epochs=0,
            epoch_length=0,
            eval_samples=0,  # Ignore eval. Just do this to remove warnings.
            discount=discount,
        )

        obs = np.array([[0., 1., 1., -11.],
                        [5., 10., 10., -10.]])

        # loss = -1/N sum_i Q(s_i, u(s_i))
        #      = -1/2 * {(Q([0,1,1,-11], u([0,1,1,-11]))
        #                + Q([5,10,10,-10], u([5,10,10,-10]))}
        #      = -1/2 * {Q([0,1,1,-11], -9)) + Q([5,10,10,-10], 15))}
        #      = -1/2 * (-18 + 30)
        #      = -6
        feed_dict = algo._actor_feed_dict(obs)
        actual = algo.sess.run(algo.actor_surrogate_loss, feed_dict=feed_dict)
        self.assertEqual(actual, -6.)
        self.assertEqual(
            np.float32,
            type(actual)
        )

    def test_actor_gradient(self):
        discount = 0.5
        algo = DDPG(
            self.env,
            self.es,
            policies.nn_policy.SumPolicy('policies', 4, 1),
            qfunctions.nn_qfunction.SumCritic('qf', 4, 1),
            n_epochs=0,
            epoch_length=0,
            eval_samples=0,  # Ignore eval. Just do this to remove warnings.
            discount=discount,
        )

        obs = np.array([[1., 1., 1., 1.],
                        [1., 1., 1., 1.]])

        # grad = -1/N sum_{i=0}^N * dQ/da * da/dtheta
        #      = -1/2 sum_{i=0}^1 * 1 * [1,1,1,1]
        #      = - [1,1,1,1]
        feed_dict = algo._actor_feed_dict(obs)
        loss_grad_ops = tf.gradients(
                algo.actor_surrogate_loss,
                algo.policy.get_params_internal())
        actual_loss_grads = algo.sess.run(loss_grad_ops, feed_dict=feed_dict)
        actual_loss_grads_flat = np.vstack(actual_loss_grads).flatten()
        expected = [-1 * np.ones_like(v) for v in algo.policy.get_param_values()]
        self.assertTrue(
            are_np_array_lists_equal(actual_loss_grads_flat, expected)
        )

    def test_actor_gradient2(self):
        discount = 0.5
        algo = DDPG(
            self.env,
            self.es,
            policies.nn_policy.SumPolicy('policies', 4, 1),
            qfunctions.nn_qfunction.SumCritic('qf', 4, 1),
            n_epochs=0,
            epoch_length=0,
            eval_samples=0,  # Ignore eval. Just do this to remove warnings.
            discount=discount,
        )

        obs = np.array([[1., -10., 1., 2.],
                        [1., 100., 1., 2.]])

        # grad = -1/N sum_{i=0}^N * dQ/da * da/dtheta
        #      = -1/2 * 1 * [1,-10,1,2]
        #         + -1/2 * 1 * [1,100,1,2]
        #      = - [1., 45., 1., 2.]
        feed_dict = algo._actor_feed_dict(obs)
        loss_grad_ops = tf.gradients(
                algo.actor_surrogate_loss,
                algo.policy.get_params_internal())
        actual_loss_grads = algo.sess.run(loss_grad_ops, feed_dict=feed_dict)
        expected = [np.array([[-1.], [-45.], [-1.], [-2.]])]
        self.assertTrue(
            are_np_array_lists_equal(actual_loss_grads, expected)
        )

    def test_critic_copy_values_are_same(self):
        algo = DDPG(
            self.env,
            self.es,
            policies.nn_policy.SumPolicy('policies', 4, 1),
            qfunctions.nn_qfunction.SumCritic('qf', 4, 1),
            n_epochs=3,
            epoch_length=10,
            eval_samples=10,  # Ignore eval. Just do this to remove warnings.
            min_pool_size=2,
        )
        critic = algo.qf
        critic_copy = algo.critic_with_action_input
        old_params = critic.get_param_values()
        self.assertParamsEqual(critic_copy, critic)
        algo.train()
        self.assertParamsEqual(critic_copy, critic)
        self.assertFalse(are_np_array_lists_equal(old_params,
                                                  critic.get_param_values()))

    def test_only_critic_values_change(self):
        discount = 0.5
        algo = DDPG(
            self.env,
            self.es,
            policies.nn_policy.SumPolicy('policies', 4, 1),
            qfunctions.nn_qfunction.SumCritic('qf', 4, 1),
            n_epochs=0,
            epoch_length=0,
            eval_samples=0,  # Ignore eval. Just do this to remove warnings.
            discount=discount,
        )

        old_critic_values = algo.qf.get_param_values()
        old_critic_copy_values = (
            algo.critic_with_action_input.get_param_values())
        old_actor_values = algo.policy.get_param_values()
        old_target_critic_values = algo.target_qf.get_param_values()
        old_target_actor_values = algo.target_policy.get_param_values()

        rewards = np.array([3.])
        terminals = np.array([0.])
        obs = np.array([[1., 1., 1., 1.]])
        actions = np.array([[-0.5]])
        next_obs = np.array([[1., 1., 1., 1.]])
        feed_dict = algo._critic_feed_dict(rewards, terminals, obs, actions,
                                           next_obs)
        algo.sess.run(algo.train_critic_op, feed_dict=feed_dict)

        new_critic_values = algo.qf.get_param_values()
        new_critic_copy_values = (
            algo.critic_with_action_input.get_param_values())
        new_actor_values = algo.policy.get_param_values()
        new_target_critic_values = algo.target_qf.get_param_values()
        new_target_actor_values = algo.target_policy.get_param_values()

        self.assertTrue(are_np_array_lists_equal(
            old_actor_values,
            new_actor_values
        ))
        self.assertFalse(are_np_array_lists_equal(
            old_critic_values,
            new_critic_values
        ))
        self.assertFalse(are_np_array_lists_equal(
            old_critic_copy_values,
            new_critic_copy_values
        ))
        self.assertTrue(are_np_array_lists_equal(
            old_target_actor_values,
            new_target_actor_values
        ))
        self.assertTrue(are_np_array_lists_equal(
            old_target_critic_values,
            new_target_critic_values
        ))
        self.assertParamsEqual(algo.critic_with_action_input, algo.qf)

    def test_only_actor_values_change(self):
        discount = 0.5
        algo = DDPG(
            self.env,
            self.es,
            policies.nn_policy.SumPolicy('policies', 4, 1),
            qfunctions.nn_qfunction.SumCritic('qf', 4, 1),
            n_epochs=0,
            epoch_length=0,
            eval_samples=0,  # Ignore eval. Just do this to remove warnings.
            discount=discount,
        )
        old_critic_values = algo.qf.get_param_values()
        old_critic_copy_values = (
            algo.critic_with_action_input.get_param_values())
        old_actor_values = algo.policy.get_param_values()
        old_target_critic_values = algo.target_qf.get_param_values()
        old_target_actor_values = algo.target_policy.get_param_values()

        obs = np.array([[1., 1., 1., 1.]])
        feed_dict = algo._actor_feed_dict(obs)
        algo.sess.run(algo.train_actor_op, feed_dict=feed_dict)

        new_critic_values = algo.qf.get_param_values()
        new_critic_copy_values = (
            algo.critic_with_action_input.get_param_values())
        new_actor_values = algo.policy.get_param_values()
        new_target_critic_values = algo.target_qf.get_param_values()
        new_target_actor_values = algo.target_policy.get_param_values()

        self.assertFalse(are_np_array_lists_equal(
            old_actor_values,
            new_actor_values
        ))
        self.assertTrue(are_np_array_lists_equal(
            old_critic_values,
            new_critic_values
        ))
        self.assertTrue(are_np_array_lists_equal(
            old_critic_copy_values,
            new_critic_copy_values
        ))
        self.assertTrue(are_np_array_lists_equal(
            old_target_actor_values,
            new_target_actor_values
        ))
        self.assertTrue(are_np_array_lists_equal(
            old_target_critic_values,
            new_target_critic_values
        ))

if __name__ == '__main__':
    unittest.main()
