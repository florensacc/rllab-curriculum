


from sandbox.rocky.hrl.envs.compound_action_sequence_env import CompoundActionSequenceEnv
# from sandbox.rocky.hrl.algos.imitation_algos import ImitationAlgo
from sandbox.rocky.hrl.algos.theano_imitation_algos import ImitationAlgo
from rllab.misc.instrument import stub, run_experiment_lite

stub(globals())

from rllab.misc.instrument import VariantGenerator

vg = VariantGenerator()
vg.add("seed", [11, 111, 211, 311, 411])
vg.add("learning_rate", [1e-3, 7.5e-4, 5e-4, 2.5e-4, 1e-4])
vg.add("bottleneck_coeff", [0.1])#1., 10., 0.1, 0.01, 0.001, 0.])

variants = vg.variants()

print("#Experiments:", len(variants))

for v in variants:
    algo = ImitationAlgo(
        learning_rate=v["learning_rate"],
        bottleneck_coeff=v["bottleneck_coeff"],
    )
    run_experiment_lite(
        algo.train(),
        exp_prefix="seq_grid_imitation_2",
        seed=v["seed"],
        variant=v,
        mode="local"
    )
    break
