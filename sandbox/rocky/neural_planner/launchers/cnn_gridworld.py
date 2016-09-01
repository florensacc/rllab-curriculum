from __future__ import print_function
from __future__ import absolute_import

from sandbox.rocky.neural_planner.gridworld_benchmark import GridworldBenchmark
from rllab.misc.instrument import stub, run_experiment_lite

stub(globals())

run_experiment_lite(
    GridworldBenchmark(
        n_maps=100,
    ).train(),
)