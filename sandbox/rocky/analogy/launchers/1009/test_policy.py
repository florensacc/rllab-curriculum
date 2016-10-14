def test_policy():
    from sandbox.rocky.analogy.networks.simple_particle.attention import Net
    # from sandbox.rocky.analogy.networks.simple_particle.double_rnn import Net
    from sandbox.rocky.analogy.policies.modular_analogy_policy import ModularAnalogyPolicy
    from rllab.envs.base import EnvSpec
    from sandbox.rocky.tf.spaces.box import Box
    import tensorflow as tf
    import numpy as np

    obs_dim = 10
    action_dim = 3
    T = 100

    observation_space = Box(low=-1, high=1, shape=(obs_dim,))
    action_space = Box(low=-1, high=1, shape=(action_dim,))

    policy = ModularAnalogyPolicy(
        env_spec=EnvSpec(
            observation_space=observation_space,
            action_space=action_space,
        ),
        name="policy",
        net=Net(obs_type='full_state')
    )

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        # Test that the result of computing all the actions at once agrees with computing actions sequentially

        demo_path = dict(
            observations=np.random.uniform(low=-1, high=1, size=(T, obs_dim)),
            actions=np.random.uniform(low=-1, high=1, size=(T, action_dim))
        )
        analogy_obs = np.random.uniform(low=-1, high=1, size=(T, obs_dim))
        analogy_actions = []
        policy.apply_demo(demo_path)
        policy.reset()
        for t in range(T):
            analogy_actions.append(policy.get_action(analogy_obs[t])[0])

        demo_obs_var = observation_space.new_tensor_variable(name="demo_obs", extra_dims=2)
        demo_actions_var = action_space.new_tensor_variable(name="demo_actions", extra_dims=2)
        demo_valids_var = tf.ones(tf.pack([tf.shape(demo_obs_var)[0], tf.shape(demo_obs_var)[1]]))
        analogy_obs_var = observation_space.new_tensor_variable(name="analogy_obs", extra_dims=2)

        analogy_actions_var = policy.action_sym(obs_var=analogy_obs_var, state_info_vars=dict(
            demo_obs=demo_obs_var, demo_actions=demo_actions_var, demo_valids=demo_valids_var
        ))

        analogy_actions_2 = sess.run(
            analogy_actions_var,
            feed_dict={
                demo_obs_var: [demo_path["observations"]],
                demo_actions_var: [demo_path["actions"]],
                analogy_obs_var: [analogy_obs],
            }
        )[0]

        np.testing.assert_allclose(analogy_actions_2, analogy_actions, atol=1e-6)

        # Test that the policy is not affected by entries marked as invalid

        analogy_actions_3 = sess.run(
            analogy_actions_var,
            feed_dict={
                demo_obs_var: [np.concatenate([demo_path["observations"]]*2, axis=0)],
                demo_actions_var: [np.concatenate([demo_path["actions"]]*2, axis=0)],
                analogy_obs_var: [np.concatenate([analogy_obs], axis=0)],
                demo_valids_var: [np.concatenate([np.ones(T), np.zeros(T)])],
            }
        )[0]

        np.testing.assert_allclose(analogy_actions_2, analogy_actions_3, atol=1e-6)

if __name__ == "__main__":
    test_policy()
