from __future__ import absolute_import
from __future__ import print_function

from rllab.misc.instrument import run_experiment_lite

"""
mutual information bonus + deterministic bottleneck + extra entropy bonus
"""

from sandbox.rocky.hrl_imitation.algos.fixed_clock_imitation3 import FixedClockImitation

from rllab.misc.instrument import VariantGenerator

vg = VariantGenerator()
vg.add("seed", [11, 111, 211, 311, 411])
vg.add("learning_rate", [1e-3])

variants = vg.variants()

print("#Experiments:", len(variants))

for v in variants:
    algo = FixedClockImitation(
        learning_rate=v["learning_rate"],
        n_sweep_per_epoch=5,
    )
    run_experiment_lite(
        algo.train(),
        exp_prefix="seq_grid_imitation_6",
        seed=v["seed"],
        variant=v,
        mode="local"
    )
    break
