from sandbox.rocky.hogwild.async_ddpg import AsyncDDPG
from rllab.algos.ddpg import DDPG
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.misc import instrument
import numpy as np

stub(globals())

ASYNC = True

"""
Best parameters:

Cart pole: scale reward 0.01, qf lr 0.001, policy lr 0.001. need about 100000 samples
Swimmer: scale reward 1.0, qf lr 0.001, policy lr 0.001. need about 200000 samples
Double Pendulum: scale reward 0.1, qf lr 0.0005, policy lr 0.0005. need about 350000 samples
"""


# Naively applying brute-force hogwild did not work


#settings = [
#    dict(qf_learning_rate=1e-4, policy_learning_rate=1e-5, use_replay_pool=True, batch_size=32, soft_target_tau=),
#    dict(qf_learning_rate=1e-5, policy_learning_rate=1e-5, use_replay_pool=True, batch_size=32, ),
#    dict(qf_learning_rate=1e-4, policy_learning_rate=1e-4, use_replay_pool=True, batch_size=32, ),
#    dict(qf_learning_rate=1e-3, policy_learning_rate=1e-4, use_replay_pool=True, batch_size=32, ),
#    dict(qf_learning_rate=1e-3, policy_learning_rate=1e-3, use_replay_pool=True, batch_size=32, ),
#
#]
if ASYNC:
    env = normalize(SwimmerEnv())#CartpoleEnv())#DoublePendulumEnv())
    vg = instrument.VariantGenerator()
    vg.add("soft_target_tau", [1e-3, 5e-4, 1e-4])
    vg.add("qf_learning_rate", [1e-3, 1e-4, 1e-5])
    vg.add("policy_lr_ratio", [1, 0.1])
    vg.add("scale_reward", [0.1, 1])
    vg.add("use_replay_pool", [False])#True, False])
    vg.add("batch_size", [5, 10, 15, 20, 32])
    vg.add("hidden_sizes", [(400, 300)])#(32, 32), (400, 300)])
    vg.add("seed", [11, 111, 211, 311, 411])
    vg.add("n_workers", [16])#20])#4, 10, 16, 20])
    vg.add("qf_weight_decay", [1e-2, 0, 1e-3, 1e-4])

    print "#Experiments:", len(vg.variants())
    variants = vg.variants()
    np.random.shuffle(variants)
    for variant in variants:
        #if variant["use_replay_pool"]:
        #    variant["batch_size"] = 32
        #else:
        #    variant["batch_size"] = 5

        policy = DeterministicMLPPolicy(env_spec=env.spec, hidden_sizes=variant["hidden_sizes"])
        qf = ContinuousMLPQFunction(env_spec=env.spec, hidden_sizes=variant["hidden_sizes"])

        variant["policy_learning_rate"] = variant["qf_learning_rate"] * variant["policy_lr_ratio"]
        es = OUStrategy(env_spec=env.spec)
        #worker_es = []
        #for i in xrange(n_workers):
        #    worker_es.append(OUStrategy(env_spec=env.spec, sigma=(i+1) * 1.0 / n_workers))#0.3))
        algo = AsyncDDPG(
            env=env, policy=policy, qf=qf, n_workers=variant["n_workers"], es=es, scale_reward=variant["scale_reward"],
            qf_learning_rate=variant["qf_learning_rate"], max_path_length=500,
            policy_learning_rate=variant["policy_learning_rate"], max_samples=5000000, use_replay_pool=variant["use_replay_pool"],
            batch_size=variant["batch_size"], qf_weight_decay=variant["qf_weight_decay"], policy_weight_decay=1e-7, evaluate_policy=True, min_eval_interval=10000,
            target_update_method='soft', hard_target_interval=40000, soft_target_tau=variant["soft_target_tau"], sync_mode="none",
        )
        #for seed in [11, 21, 31, 41, 51]:
        num_threads = int(np.floor(1.0 * 36 / variant["n_workers"]))
        print("num threads: %d" % num_threads)
        run_experiment_lite(
            algo.train(),
            exp_prefix="async_ddpg_swimmer_extreme",
            seed=variant["seed"],
            mode="local",
            env=dict(OMP_NUM_THREADS=str(num_threads))
        )
        #break
else:
    for seed in [11, 21, 31]:
        for env in map(normalize, [DoublePendulumEnv(), SwimmerEnv(), CartpoleEnv()]):
            policy = DeterministicMLPPolicy(env_spec=env.spec)
            qf = ContinuousMLPQFunction(env_spec=env.spec)
            es = OUStrategy(env_spec=env.spec)

            for scale_reward in [1.0, 0.1, 0.01]:
                for qf_lr in [1e-3, 5e-4, 1e-4, 5e-4, 1e-5]:
                    for policy_lr_mult in [1, 10, 50]:
                        algo = DDPG(env=env, policy=policy, qf=qf, es=es, scale_reward=scale_reward, qf_learning_rate=qf_lr,
                                    policy_learning_rate=qf_lr / policy_lr_mult, max_path_length=500, n_epochs=1000)
                        run_experiment_lite(
                            algo.train(),
                            exp_prefix="async_ddpg",
                            seed=seed,
                            mode="local",
                        )

