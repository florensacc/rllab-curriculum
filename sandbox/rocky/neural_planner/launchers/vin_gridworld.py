from __future__ import print_function
from __future__ import absolute_import

from sandbox.rocky.neural_planner.gridworld_benchmark import GridworldBenchmark, CNN, VIN
from rllab.misc.instrument import stub, run_experiment_lite
from rllab import config

stub(globals())

from rllab.misc.instrument import VariantGenerator, variant

vg = VariantGenerator()
vg.add("network", ["vin"])#, "cnn"])
vg.add("seed", [21])#11, 21, 31, 41, 51])
vg.add("shape", [(28, 28)])#16, 16)])
vg.add("n_iter", [36])#lambda network: [10, 20, 36] if network == "vin" else [None])

variants = vg.variants()
print("#Experiments: %d" % len(variants))

KUBE_DEFAULT_RESOURCES = {
    "requests": {
        "cpu": 16 * 0.75,
    },
}

for v in variants:
    if v["network"] == "cnn":
        network = CNN()
    elif v["network"] == "vin":
        network = VIN(v["n_iter"])
    else:
        raise NotImplementedError
    network = CNN()
    task = GridworldBenchmark(
        n_maps=5000,
        network=network,
        shape=v["shape"],
        max_n_obstacles=10,
        n_epochs=120,
        train_ratio=0.95,
        learning_rate=0.001,
        # lr_schedule=[(0.01, 30), (0.005, 30), (0.002, 30), (0.001, 30)]
    )
    run_experiment_lite(
        task.train(),
        n_parallel=0,
        seed=v["seed"],
        variant=v,
        exp_prefix="vin_gridworld",
        mode="local"
    )
    break
