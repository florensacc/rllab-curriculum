


from rllab.misc.instrument import stub, run_experiment_lite

"""
Use natural gradient
Also got stuck at the same local optima. Hence, I don't expect the annealing technique to work
"""

from sandbox.rocky.hrl_imitation.algos.fixed_clock_imitation12 import FixedClockImitation, SeqGridPolicyModule

stub(globals())
from rllab.misc.instrument import VariantGenerator

# import tensorflow as tf
# from rllab.misc import logger


# seed 11: doesn't work
vg = VariantGenerator()
vg.add("seed", [x * 100 + 11 for x in range(10)])  # 911])#11, 111, 211, 311, 411, 511, 611, 711, 811, 911])
vg.add("learning_rate", [1e-3])
vg.add("mi_coeff", [0.])#, 0.001, 0.01, 0.1, 1.])
vg.add("low_policy_obs", ['partial'])#'full', 'partial'])

variants = vg.variants()

print("#Experiments:", len(variants))

iters = []
for v in variants:
    algo = FixedClockImitation(
        policy_module=SeqGridPolicyModule(low_policy_obs=v['low_policy_obs']),
        learning_rate=v["learning_rate"],
        mi_coeff=v["mi_coeff"],
        n_epochs=500,
        bottleneck_dim=100,
        batch_size=None,
    )
    run_experiment_lite(
        algo.train(),
        exp_prefix="seq_grid_imitation_25",
        seed=v["seed"],
        variant=v,
        mode="local",
        snapshot_mode="last",
    )
    break
