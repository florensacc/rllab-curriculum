from nose.tools import assert_raises
import numpy as np


def test_compound_action_sequence_mdp():
    from rllab.mdp.grid_world_mdp import GridWorldMDP
    from rllab.mdp.compound_action_sequence_mdp import CompoundActionSequenceMDP
    from rllab.misc import special
    mdp = GridWorldMDP(desc=[
        "SFFF",
        "FFFF",
        "FFFF",
        "FFFG",
    ])
    # this should succeed
    CompoundActionSequenceMDP(mdp=mdp, action_map=[[0, 0, 0], [1], [2, 2, 2], [3, 3]])
    assert_raises(AssertionError, CompoundActionSequenceMDP, mdp=mdp, action_map=[])
    assert_raises(AssertionError, CompoundActionSequenceMDP, mdp=mdp, action_map=[[0, 0], [0], [1], [2]])
    assert_raises(AssertionError, CompoundActionSequenceMDP, mdp=mdp, action_map=[[], [0], [1], [2]])

    compound_mdp = CompoundActionSequenceMDP(mdp=mdp, action_map=[[0, 0, 0], [0, 1, 2], [2, 2, 2], [3, 3, 3]])

    def check_action_sequence(actions, outcomes):
        compound_mdp.reset()
        for action, outcome in zip(actions, outcomes):
            obs = compound_mdp.step(special.to_onehot(action, 4))[0]
            np.testing.assert_array_equal(obs, special.to_onehot(outcome, 16))

    check_action_sequence([0, 1, 2], [0, 0, 4])
    check_action_sequence([0, 0, 0], [0, 0, 0])
    check_action_sequence([0, 1, 1, 2, 2, 2], [0, 0, 0, 0, 0, 1])
