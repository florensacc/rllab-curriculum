


from sandbox.rocky.neural_planner.gridworld_benchmark import GridworldBenchmark, CNN, VIN, VIN1, VINMulti
from rllab.misc.instrument import stub, run_experiment_lite
from rllab import config

stub(globals())

from rllab.misc.instrument import VariantGenerator, variant

vg = VariantGenerator()
vg.add("seed", [11])#, 21, 31, 41, 51])
vg.add("shape", [(29, 29)])#16, 16)])#17, 17)])#28, 28)])#16, 16)])#, (28, 28)])
vg.add("n_iter", range(36))
# vg.add("n_q_filters", [10, 20, 30])
# vg.add("has_nonlinearity", [True, False])
# vg.add("untie_weights", [True, False])

variants = vg.variants()#randomized=True)
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
    # network = CNN(n_iter=v["n_iter"])
    network = VIN(
        n_iter=v["n_iter"],#4,#20
        n_q_filters=20,
        has_nonlinearity=False,
        untie_weights=False,
    )
    #     n_iter=v["n_iter"],
    #     n_q_filters=v["n_q_filters"],
    #     has_nonlinearity=v["has_nonlinearity"],
    #     untie_weights=v["untie_weights"],
    # )
    task = GridworldBenchmark(
        n_maps=5000,
        network=network,
        shape=v["shape"],
        max_n_obstacles=10,
        n_epochs=40,
        train_ratio=0.95,
        learning_rate=0.001,
        # learning_rate=0.01,
        batch_size=64,
        gradient_noise_scale=0,#.0001,
    )
    run_experiment_lite(
        task.train(),
        n_parallel=0,
        seed=v["seed"],
        variant=v,
        exp_prefix="vin_iters",
        mode="local",
        # env=dict(THEANO_FLAGS="optimizer=fast_compile,device=cpu"),
    )
    # break
