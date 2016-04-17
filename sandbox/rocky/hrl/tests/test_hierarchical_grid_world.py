from __future__ import print_function
from __future__ import absolute_import
import numpy as np
from rllab.spaces.discrete import Discrete
from rllab.spaces.product import Product
# from nose2.tools import

from nose2.tools import such

from sandbox.rocky.hrl.hierarchical_grid_world_env import HierarchicalGridWorldEnv
from sandbox.rocky.hrl.hierarchical_grid_world_env import expand_grid
from sandbox.rocky.hrl.subgoal_policy import SubgoalPolicy
from rllab.envs.grid_world_env import GridWorldEnv
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
import time

with such.A("expand_grid") as it:
    @it.should("work")
    def test_expand_grid():
        high_grid = [
            "SFF",
            "FWH",
            "FFG"
        ]
        low_grid = [
            "SFF",
            "FFF",
            "FFG"
        ]
        total_grid = expand_grid(high_grid, low_grid)
        it.assertEqual(total_grid.shape, (9, 9))
        it.assertEqual(np.sum(total_grid == 'F'), 8 + 9 + 9 + 9 + 9 + 9 + 8)
        it.assertEqual(np.sum(total_grid == 'S'), 1)
        it.assertEqual(np.sum(total_grid == 'G'), 1)
        it.assertEqual(np.sum(total_grid == 'W'), 9)
        it.assertEqual(np.sum(total_grid == 'H'), 9)
        it.assertEqual(total_grid[0, 0], 'S')
        it.assertEqual(total_grid[8, 8], 'G')

it.createTests(globals())

with such.A("hierarchical grid world") as it:
    @it.should("work")
    def test_hierarchical_grid_world():
        high_grid = [
            "SFF",
            "FWH",
            "FFG"
        ]
        low_grid = [
            "SFF",
            "FFF",
            "FFG"
        ]
        hier_grid_world = HierarchicalGridWorldEnv(high_grid, low_grid)
        it.assertEqual(hier_grid_world.observation_space, Product(Discrete(9), Discrete(9)))
        it.assertEqual(hier_grid_world.action_space, Discrete(4))
        state = hier_grid_world.reset()
        it.assertEqual(state, (0, 0))
        right_action = GridWorldEnv.action_from_direction("right")
        it.assertEqual(hier_grid_world.step(right_action)[0], (0, 1))
        it.assertEqual(hier_grid_world.step(right_action)[0], (0, 2))
        it.assertEqual(hier_grid_world.step(right_action)[0], (1, 0))

it.createTests(globals())

with such.A("hierarchical grid world analyzer") as it:
    @it.should("compute basic policy-independent quantities")
    def test_hierarchical_grid_world_analyzer_policy_independent():
        high_grid = [
            "SF",
            "FF"
        ]
        low_grid = [
            "SF",
            "FF"
        ]
        hier_grid_world = HierarchicalGridWorldEnv(high_grid, low_grid)
        act = hier_grid_world.flat_env.action_from_direction
        analyzer = hier_grid_world.analyzer
        posterior_seqs = analyzer.compute_posterior_sequences(1)
        it.assertEqual(len(posterior_seqs[(0, 1)]), 1)
        it.assertEqual(
            posterior_seqs[(0, 1)][0],
            ((), (act("right"),), 1.0)
        )
        it.assertEqual(len(posterior_seqs[(0, 0)]), 2)
        it.assertEqual(
            set(posterior_seqs[(0, 0)]),
            set([
                ((), (act("left"),), 0.5),
                ((), (act("up"),), 0.5),
            ])
        )
        posterior_seqs2 = analyzer.compute_posterior_sequences(2)
        it.assertEqual(len(posterior_seqs2[(0, 5)]), 2)
        it.assertEqual(
            set(posterior_seqs2[(0, 5)]),
            set([
                ((1,), (act("right"), act("down")), 0.5),
                ((4,), (act("down"), act("right")), 0.5),
            ])
        )
        it.assertEqual(
            analyzer.get_sequence_transition_probability(0, (1, 5), (act("right"), act("down"))),
            1.0
        )
        it.assertEqual(
            analyzer.get_sequence_transition_probability(0, (4, 5), (act("down"), act("right"))),
            1.0
        )

        for state in xrange(analyzer.n_states):
            prob = 0.
            for next_state in xrange(analyzer.n_states):
                for state_seq, action_seq, _ in analyzer.get_posterior_sequences(2, state, next_state):
                    prob += analyzer.get_sequence_transition_probability(state, state_seq + (next_state,), action_seq)
            # assume a uniform prior over the actions
            it.assertEqual(prob / 16., 1.)


    @it.should("correctly convert indices")
    def test_analyzer_convert_indices():
        high_grid = [
            "SF",
            "FF"
        ]
        low_grid = [
            "SF",
            "FF"
        ]
        hier_grid_world = HierarchicalGridWorldEnv(high_grid, low_grid)
        act = hier_grid_world.flat_env.action_from_direction
        analyzer = hier_grid_world.analyzer
        hier_grid_world.reset()
        next_obs, _, _, _ = hier_grid_world.step(act("right"))
        it.assertEqual(next_obs, (0, 1))
        it.assertEqual(analyzer.get_int_state_from_obs(next_obs), 1)
        next_obs, _, _, _ = hier_grid_world.step(act("right"))
        it.assertEqual(next_obs, (1, 0))
        it.assertEqual(analyzer.get_int_state_from_obs(next_obs), 2)


    @it.should("compute policy-dependent quantities")
    def test_hierarchical_grid_world_analyzer_with_policy():
        high_grid = [
            "SF",
            "FF"
        ]
        low_grid = [
            "SF",
            "FF"
        ]
        hier_grid_world = HierarchicalGridWorldEnv(high_grid, low_grid)
        start_time = time.time()
        policy = SubgoalPolicy(
            env_spec=hier_grid_world.spec,
            high_policy_cls=CategoricalMLPPolicy,
            high_policy_args=dict(),
            low_policy_cls=CategoricalMLPPolicy,
            low_policy_args=dict(),
            subgoal_interval=3,
            subgoal_space=Discrete(4),
        )
        print("compiling policy took %s" % (time.time() - start_time))
        start_time = time.time()
        hier_grid_world.analyzer.set_policy(policy)
        goal_probs = hier_grid_world.analyzer.compute_goal_transition_probabilities()
        print("computing goal probs took %s" % (time.time() - start_time))
        np.testing.assert_allclose(np.sum(goal_probs, axis=-1), 1.)

it.createTests(globals())
