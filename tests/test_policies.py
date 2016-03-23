# def test_categorical_gru_policy():
#     from rllab.policy.categorical_gru_policy import CategoricalGRUPolicy
#     from rllab.env.grid_world_env import GridWorldEnv
#     mdp = GridWorldMDP()
#     policy = CategoricalGRUPolicy(
#         mdp
#     )
#     policy.get_action(mdp.reset())


def test_categorical_mlp_policy():
    from rllab.policy.categorical_mlp_policy import CategoricalMLPPolicy
    from rllab.env.grid_world_env import GridWorldEnv
    env = GridWorldEnv()
    policy = CategoricalMLPPolicy(
        env_spec=env.spec,
    )
    action = policy.get_action(env.reset())[0]
    assert env.observation_space.contains(action)
