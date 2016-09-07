


from rllab.misc.instrument import stub, run_experiment_lite

"""
Batch optimization. Optimize I(a;g|z). But with smaller networks
"""

from sandbox.rocky.hrl_imitation.algos.fixed_clock_imitation9 import FixedClockImitation, SeqGridPolicyModule

stub(globals())
from rllab.misc.instrument import VariantGenerator

# import tensorflow as tf
# from rllab.misc import logger


# seed 11: doesn't work
vg = VariantGenerator()
vg.add("seed", [x * 100 + 11 for x in range(10)])  # 911])#11, 111, 211, 311, 411, 511, 611, 711, 811, 911])
vg.add("learning_rate", [1e-3])
vg.add("mi_coeff", [0., 0.001, 0.01, 0.1, 1.])
vg.add("low_policy_obs", ['full'])#, 'partial'])

variants = vg.variants()

print("#Experiments:", len(variants))

iters = []
for v in variants:
    algo = FixedClockImitation(
        policy_module=SeqGridPolicyModule(low_policy_obs=v['low_policy_obs']),
        learning_rate=v["learning_rate"],
        mi_coeff=v["mi_coeff"],
        n_epochs=500,
        batch_size=None,
    )
    run_experiment_lite(
        algo.train(),
        exp_prefix="seq_grid_imitation_22",
        seed=v["seed"],
        variant=v,
        mode="lab_kube",
        snapshot_mode="last",
    )
    # break
