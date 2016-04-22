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

stub(globals())

ASYNC = True

"""
Best parameters:

Cart pole: scale reward 0.01, qf lr 0.001, policy lr 0.001. need about 100000 samples
Swimmer: scale reward 1.0, qf lr 0.001, policy lr 0.001. need about 200000 samples
Double Pendulum: scale reward 0.1, qf lr 0.0005, policy lr 0.0005. need about 350000 samples
"""


# Naively applying brute-force hogwild did not work

if ASYNC:
    env = normalize(SwimmerEnv())#CartpoleEnv())#DoublePendulumEnv())
    policy = DeterministicMLPPolicy(env_spec=env.spec, hidden_sizes=(32, 32))#, hidden_sizes=(400, 300))
    qf = ContinuousMLPQFunction(env_spec=env.spec, hidden_sizes=(32, 32))#, hidden_sizes=(400, 300))
    es = OUStrategy(env_spec=env.spec)
    # worker_es = [
    #     OUStrategy(env_spec=env.spec, sigma=0.3),
    #     OUStrategy(env_spec=env.spec, sigma=0.2),
    #     OUStrategy(env_spec=env.spec, sigma=0.1),
    #     OUStrategy(env_spec=env.spec, sigma=0.4),
    # ]
    algo = AsyncDDPG(
        env=env, policy=policy, qf=qf, n_workers=4, es=es, scale_reward=1, qf_learning_rate=1e-3,
        max_path_length=500, policy_learning_rate=1e-3, max_samples=10000000, use_replay_pool=True, batch_size=32,
        qf_weight_decay=1e-7, policy_weight_decay=1e-7, evaluate_policy=True, min_eval_interval=10000,
        soft_target_tau=1e-3,
        sync_mode="none",
    )
    for seed in [11, 21, 31, 41, 51]:
        run_experiment_lite(
            algo.train(),
            exp_prefix="async_ddpg_swimmer",
            seed=seed,
            mode="local",
            env=dict(OMP_NUM_THREADS="1")
        )
        break
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

