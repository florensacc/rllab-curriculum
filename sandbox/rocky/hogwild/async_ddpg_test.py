from sandbox.rocky.hogwild.async_ddpg import AsyncDDPG
from rllab.algos.ddpg import DDPG
from rllab.envs.box2d.cartpole_swingup_env import CartpoleSwingupEnv
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv
from rllab.envs.mujoco.inverted_double_pendulum_env import InvertedDoublePendulumEnv
from rllab.envs.mujoco.ant_env import AntEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.misc import instrument
from rllab import config
import numpy as np
import sys

stub(globals())

ASYNC = True

"""
Best parameters:

Cart pole: scale reward 0.01, qf lr 0.001, policy lr 0.001. need about 100000 samples
Swimmer: scale reward 1.0, qf lr 0.001, policy lr 0.001. need about 200000 samples
Double Pendulum: scale reward 0.1, qf lr 0.0005, policy lr 0.0005. need about 350000 samples
"""

# Naively applying brute-force hogwild did not work


# settings = [
#    dict(qf_learning_rate=1e-4, policy_learning_rate=1e-5, use_replay_pool=True, batch_size=32, soft_target_tau=),
#    dict(qf_learning_rate=1e-5, policy_learning_rate=1e-5, use_replay_pool=True, batch_size=32, ),
#    dict(qf_learning_rate=1e-4, policy_learning_rate=1e-4, use_replay_pool=True, batch_size=32, ),
#    dict(qf_learning_rate=1e-3, policy_learning_rate=1e-4, use_replay_pool=True, batch_size=32, ),
#    dict(qf_learning_rate=1e-3, policy_learning_rate=1e-3, use_replay_pool=True, batch_size=32, ),
#
# ]



inverted_double_pendulum_env = normalize(InvertedDoublePendulumEnv())
swimmer_env = normalize(SwimmerEnv())
cartpole_swingup_env = normalize(CartpoleSwingupEnv())
ant_env = normalize(AntEnv())

reward_scaling = {
    # inverted_double_pendulum_env: 0.01,
    # swimmer_env: 1,
    # cartpole_swingup_env: 0.1,
    ant_env: 0.01,
}

if ASYNC:
    vg = instrument.VariantGenerator()

    vg.add("env", reward_scaling.keys())
    vg.add("soft_target_tau", [1e-3])  # , 1e-4])
    vg.add("n_workers", [1, 4])#, 8, 16])
    vg.add("qf_learning_rate", [1e-4, 1e-5])#lambda n_workers: [1e-3, 1e-4] if n_workers <= 4 else [1e-4, 1e-5])
    vg.add("policy_lr_ratio", [0.1])
    vg.add("scale_reward", lambda env: [reward_scaling[env]])
    vg.add("use_replay_pool", [True])  # False, True])
    vg.add("batch_size", [32])  # lambda use_replay_pool: [32] if use_replay_pool else [4])  # , 16, 32])
    vg.add("hidden_sizes", [(400, 300)])
    vg.add("seed", [511, 611, 711, 811, 911])  # 11, 111, 211, 311, 411])
    vg.add("qf_weight_decay", [1e-7])
    vg.add("target_update_method", ['soft'])
    vg.add("hard_target_interval", [40000])

    print "#Experiments:", len(vg.variants())
    variants = vg.variants()

    config.AWS_INSTANCE_TYPE = 'c4.4xlarge'
    config.AWS_SPOT_PRICE = '2.0'

    # total_machines = 160
    #
    # runs_per_machine = int(np.ceil(len(variants) * 1.0 / total_machines))
    # print("Runs per machine: %d" % runs_per_machine)

    regions = [
        "us-west-1",
        "us-east-1"
    ]

    zones = {
        "us-west-1": [
            "us-west-1a",
            "us-west-1c",
        ],
        "us-east-1": [
            "us-east-1a",
            "us-east-1c",
            "us-east-1d",
            # "us-east-1e",
        ]
    }

    ami_ids = {
        "us-west-1": "ami-0c78066c",
        "us-east-1": "ami-67c5d00d",
    }

    key_names = {
        "us-west-1": "research_north_california",
        "us-east-1": "research_virginia",
    }

    # rest_ids = [
    #     82, 83, 88, 89, 92, 99, 108
    #     # 84,
    #     # 87,
    #     # 90,
    #     # 96,
    #     # 98,
    #     # 103,
    #     # 104,
    #     # 105,
    #     # 107
    # ]
    #
    # rest_ids = [x - 1 for x in rest_ids]
    #
    # variants = [variants[i] for i in rest_ids]

    n_zones = sum(map(len, zones.values()))

    zones = sum(zones.values(), [])

    runs_per_zone = int(np.ceil(len(variants) * 1.0 / n_zones))

    print("runs per zone: %d" % runs_per_zone)

    for zone, zone_start_idx in zip(zones, range(0, len(variants), runs_per_zone)):

        region = zone[:-1]

        # 6 zones, 160 jobs = around 30 jobs per zone

        for variant_idx in xrange(zone_start_idx, zone_start_idx + runs_per_zone):

            if variant_idx >= len(variants):
                sys.exit(0)

            print("*************************************************************")
            print("*************************************************************")
            print("Launching task #%d" % variant_idx)
            print("*************************************************************")
            print("*************************************************************")

            variant = variants[variant_idx]

            # slice_variants = variants[start_idx:start_idx + runs_per_machine]

            # tasks = []

            # for variant in slice_variants:
            env = variant["env"]
            policy = DeterministicMLPPolicy(env_spec=env.spec, hidden_sizes=variant["hidden_sizes"])
            qf = ContinuousMLPQFunction(env_spec=env.spec, hidden_sizes=variant["hidden_sizes"])
            variant["policy_learning_rate"] = variant["qf_learning_rate"] * variant["policy_lr_ratio"]
            es = OUStrategy(env_spec=env.spec)
            max_samples = 4000000 #max(2000000, variant["n_workers"] * 1000000)
            algo = AsyncDDPG(
                env=env, policy=policy, qf=qf, n_workers=variant["n_workers"], es=es,
                scale_reward=variant["scale_reward"],
                qf_learning_rate=variant["qf_learning_rate"], max_path_length=500,
                policy_learning_rate=variant["policy_learning_rate"], max_samples=max_samples,
                use_replay_pool=variant["use_replay_pool"], batch_size=variant["batch_size"],
                qf_weight_decay=variant["qf_weight_decay"], policy_weight_decay=1e-7, evaluate_policy=True,
                min_eval_interval=10000,
                target_update_method=variant["target_update_method"],
                hard_target_interval=variant["hard_target_interval"],
                soft_target_tau=variant["soft_target_tau"],
                sync_mode="none",
            )
            num_threads = int(np.floor(1.0 * 16 / variant["n_workers"]))
            task = dict(
                stub_method_call=algo.train(),
                seed=variant["seed"],
                env=dict(OMP_NUM_THREADS=str(num_threads))
            )
            config.AWS_REGION_NAME = region
            config.AWS_KEY_NAME = key_names[region]
            run_experiment_lite(
                exp_prefix="async_ddpg_ant_only",
                mode="ec2",
                batch_tasks=[task],  # tasks,
                terminate_machine=True,
                aws_config=dict(
                    placement=dict(
                        AvailabilityZone=zone,
                    ),
                    image_id=ami_ids[region],
                )
            )
            # break
            # sys.exit(0)
else:
    for seed in [11, 21, 31]:
        for env in map(normalize, [DoublePendulumEnv(), SwimmerEnv(), CartpoleEnv()]):
            policy = DeterministicMLPPolicy(env_spec=env.spec)
            qf = ContinuousMLPQFunction(env_spec=env.spec)
            es = OUStrategy(env_spec=env.spec)

            for scale_reward in [1.0, 0.1, 0.01]:
                for qf_lr in [1e-3, 5e-4, 1e-4, 5e-4, 1e-5]:
                    for policy_lr_mult in [1, 10, 50]:
                        algo = DDPG(env=env, policy=policy, qf=qf, es=es, scale_reward=scale_reward,
                                    qf_learning_rate=qf_lr,
                                    policy_learning_rate=qf_lr / policy_lr_mult, max_path_length=500, n_epochs=1000)
                        run_experiment_lite(
                            algo.train(),
                            exp_prefix="async_ddpg",
                            seed=seed,
                            mode="local",
                        )
