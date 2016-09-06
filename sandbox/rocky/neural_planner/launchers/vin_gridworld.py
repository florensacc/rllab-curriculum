from __future__ import print_function
from __future__ import absolute_import

from sandbox.rocky.neural_planner.gridworld_benchmark import GridworldBenchmark, CNN, VIN
from rllab.misc.instrument import stub, run_experiment_lite
from rllab import config

stub(globals())

from rllab.misc.instrument import VariantGenerator, variant

vg = VariantGenerator()
vg.add("network", ["cnn"])#, "cnn"])
vg.add("seed", [21])#11, 21, 31, 41, 51])
vg.add("shape", [(16, 16)])#16, 16)])
vg.add("n_iter", [36])#lambda network: [10, 20, 36] if network == "vin" else [None])

variants = vg.variants()
print("#Experiments: %d" % len(variants))

KUBE_DEFAULT_RESOURCES = {
    "requests": {
        "cpu": 16 * 0.75,
    },
}

for v in variants:
    # if v["network"] == "cnn":
    #     network = CNN()
    # elif v["network"] == "vin":
    #     network = VIN(v["n_iter"])
    # else:
    #     raise NotImplementedError
    network = VIN(20)#CNN()
    task = GridworldBenchmark(
        n_maps=5000,
        network=network,
        shape=v["shape"],
        max_n_obstacles=10,
        n_epochs=80,
        train_ratio=0.95,
        learning_rate=0.01,
        batch_size=128,
        gradient_noise_scale=0,#.0001,
    )
    run_experiment_lite(
        task.train(),
        n_parallel=0,
        seed=v["seed"],
        variant=v,
        exp_prefix="vin_gridworld",
        mode="local",
        # env=dict(THEANO_FLAGS="optimizer=fast_compile,device=cpu"),
    )
    break
