from rllab.misc.instrument import stub, run_experiment_lite
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.normalized_env import normalize
from sandbox.rocky.dpg.dpg import DPG
from sandbox.rocky.dpg.continuous_mlp_policy import ContinuousMLPPolicy
from sandbox.rocky.dpg.continuous_mlp_q_function import ContinuousMLPQFunction
from sandbox.rocky.dpg.ou_strategy import OUStrategy
from sandbox.rocky.dpg.gaussian_strategy import GaussianStrategy

stub(globals())

envs = [
    normalize(HalfCheetahEnv()),
    normalize(CartpoleEnv()),
]

for seed in [1]:  # , 11, 21]:
    for env in envs:
        eses = [
            OUStrategy(
                env_spec=env.spec,
                theta=0.15,
                sigma=0.2
            ),
            GaussianStrategy(
                env_spec=env.spec,

            )

        ]
        for qf_lr in [1e-3, 1e-4, 1e-5]:
            for policy_lr in [1e-4, 1e-3, 1e-5]:
                for bn in [True, False]:
                    # env = normalize(HalfCheetahEnv())#, scale_reward=0.01)
                    algo = DPG(
                        qf_learning_rate=1e-3,
                        # qf_weight_decay=0.01,
                        policy_learning_rate=1e-4,
                        max_path_length=100,
                        scale_reward=0.1,
                    )
                    policy = ContinuousMLPPolicy(
                        env_spec=env.spec,
                        hidden_sizes=(32, 32),
                        bn=bn,
                    )
                    es =
                    qf = ContinuousMLPQFunction(
                        env_spec=env.spec,
                        hidden_sizes=(32, 32),
                        bn=bn,
                    )
                    run_experiment_lite(
                        algo.train(env=env, policy=policy, qf=qf, es=es),
                        exp_prefix="old_dpg_both_search",
                        n_parallel=4,
                        snapshot_mode="last",
                    )
