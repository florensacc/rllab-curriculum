from __future__ import print_function
from __future__ import absolute_import

from sandbox.rocky.hrl.envs.compound_action_sequence_env import CompoundActionSequenceEnv
from sandbox.rocky.hrl.algos.imitation_algos import ImitationAlgo
# from rllab.misc.instrument import stub, run_experiment_lite
#
# stub(globals())

from rllab.misc.instrument import VariantGenerator

# env =

# base_map = [
#     ".....",
#     ".....",
#     ".....",
#     ".....",
#     ".....",
# ]

algo = ImitationAlgo()
algo.train()
# base_map=base_map,
# )

# all_pos = [(x, y) for x in range(5) for y in range(5)]
#
#
#
# wrapped_env = GridWorldEnv(desc=map)
# env = CompoundActionSequenceEnv(
#     wrapped_env=wrapped_env,
#     action_map=[
#         [0, 1, 1],
#         [1, 3, 3],
#         [2, 2, 0],
#         [3, 0, 2],
#     ],
#     # action_dim=2,
#     obs_include_history=True,
# )
