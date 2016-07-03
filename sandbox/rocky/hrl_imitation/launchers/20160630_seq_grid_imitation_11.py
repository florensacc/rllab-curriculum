from __future__ import absolute_import
from __future__ import print_function

from rllab.misc.instrument import stub, run_experiment_lite

"""
Debugging the simplest setting, which was found to have frequent local optima
"""

from sandbox.rocky.hrl_imitation.algos.fixed_clock_imitation3 import FixedClockImitation

stub(globals())
from rllab.misc.instrument import VariantGenerator
# import tensorflow as tf
# from rllab.misc import logger


vg = VariantGenerator()
vg.add("seed", [11, 111, 211, 311, 411, 511, 611, 711, 811, 911])
vg.add("learning_rate", [1e-3])
vg.add("mi_coeff", [0.])#, 0.001, 0.01, 0.1, 1., 10])
vg.add("ent_g_given_z_coeff", [0.])#, 0.001, 0.01, 0.1, 1., 10])

variants = vg.variants()

print("#Experiments:", len(variants))

iters = []
for v in variants:
    algo = FixedClockImitation(
        learning_rate=v["learning_rate"],
        n_sweep_per_epoch=5,
        mi_coeff=v["mi_coeff"],
        ent_g_given_z_coeff=v["ent_g_given_z_coeff"],
        n_epochs=200,
    )
    run_experiment_lite(
        algo.train(),
        exp_prefix="seq_grid_imitation_11",
        seed=v["seed"],
        variant=v,
        mode="lab_kube"
    )
