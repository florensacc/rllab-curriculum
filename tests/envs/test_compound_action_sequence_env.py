import numpy as np
from nose2 import tools


# TODO: to be fixed by rocky
def test_compound_action_sequence_env():
    from rllab.env.grid_world_env import GridWorldEnv
    from rllab.env.compound_action_sequence_env import CompoundActionSequenceEnv
    mdp = GridWorldEnv(desc=[
        "SFFF",
        "FFFF",
        "FFFF",
        "FFFG",
    ])
    # this should succeed
    CompoundActionSequenceEnv(
        wrapped_env=mdp,
        action_map=[[0, 0, 0], [1], [2, 2, 2], [3, 3]]
    )
    h = tools.such.helper
    with h.assertRaises(AssertionError):
        CompoundActionSequenceEnv(wrapped_env=mdp, action_map=[])
    with h.assertRaises(AssertionError):
        CompoundActionSequenceEnv(wrapped_env=mdp, action_map=[[0, 0], [0], [1], [2]])
    with h.assertRaises(AssertionError):
        CompoundActionSequenceEnv(wrapped_env=mdp, action_map=[[], [0], [1], [2]])

    compound_mdp = CompoundActionSequenceEnv(mdp, action_map=[[0, 0, 0], [0, 1, 2], [2, 2, 2], [3, 3, 3]])

    def check_action_sequence(actions, outcomes):
        compound_mdp.reset()
        for action, outcome in zip(actions, outcomes):
            obs = compound_mdp.step(action).observation
            np.testing.assert_array_equal(obs, outcome)

    check_action_sequence([0, 1, 2], [0, 0, 4])
    check_action_sequence([0, 0, 0], [0, 0, 0])
    check_action_sequence([0, 1, 1, 2, 2, 2], [0, 0, 0, 0, 0, 1])

