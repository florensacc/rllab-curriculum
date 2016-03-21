def test_categorical_gru_policy():
    from rllab.policy.categorical_gru_policy import CategoricalGRUPolicy
    from rllab.mdp.grid_world_mdp import GridWorldMDP
    mdp = GridWorldMDP()
    policy = CategoricalGRUPolicy(
        mdp
    )
    policy.get_action(mdp.reset())
