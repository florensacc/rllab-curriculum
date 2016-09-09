


from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.hrl_imitation.imitation_algos import ImitationAlgo

stub(globals())

from rllab.misc.instrument import VariantGenerator

vg = VariantGenerator()
vg.add("seed", [11, 111, 211, 311, 411])
vg.add("learning_rate", [1e-3, 7.5e-4, 5e-4, 2.5e-4, 1e-4])
vg.add("bottleneck_coeff", [1., 10., 0.1, 0.01, 0.001, 0.])

variants = vg.variants()

print("#Experiments:", len(variants))

for v in variants:
    algo = ImitationAlgo(
        learning_rate=v["learning_rate"],
        bottleneck_coeff=v["bottleneck_coeff"],
    )
    run_experiment_lite(
        algo.train(),
        exp_prefix="seq_grid_imitation_1",
        seed=v["seed"],
        variant=v,
        mode="lab_kube"
    )
