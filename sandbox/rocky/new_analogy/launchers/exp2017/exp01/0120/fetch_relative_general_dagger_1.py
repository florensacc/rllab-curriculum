import pyprind

from rllab.misc import logger
from rllab.sampler.utils import rollout
from sandbox.rocky.new_analogy.exp_utils import run_local_docker, run_cirrascale, run_ec2

"""
Dagger but with multiple initial positions
"""


def run_task(vv):
    from sandbox.rocky.new_analogy import fetch_utils
    from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
    import tensorflow as tf
    import numpy as np

    horizon = 500
    n_configurations = 100
    batch_size = vv["batch_size"]  # 512#4096 #512
    n_updates_per_epoch = 10000
    max_n_paths = 10000

    # n_updates_per_epoch = 100
    # n_configurations = 10

    gpr_env = fetch_utils.gpr_fetch_env(horizon=horizon)
    env = fetch_utils.fetch_env(horizon=horizon)
    demo_policy = fetch_utils.fetch_prescribed_policy(gpr_env=gpr_env)

    policy = fetch_utils.FetchWrapperPolicy(
        env_spec=env.spec,
        wrapped_policy=GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=vv["hidden_sizes"],
            hidden_nonlinearity=tf.nn.tanh,
            name="policy"
        )
    )

    paths = fetch_utils.demo_paths(
        seeds=np.random.randint(low=0, high=np.iinfo(np.int32).max, size=n_configurations),
        env=env,
        policy=demo_policy,
        noise_levels=[0., 0.001, 0.01, 0.1]
    )

    trainer = fetch_utils.bc_trainer(env=env, policy=policy, max_n_paths=max_n_paths)
    trainer.add_paths(paths)

    eval_policy = fetch_utils.DeterministicPolicy(env_spec=env.spec, wrapped_policy=policy)

    for _ in trainer.train_loop(batch_size=batch_size, n_updates_per_epoch=n_updates_per_epoch):
        logger.log("Sampling new trajectory")

        eval_paths = fetch_utils.policy_paths(
            seeds=np.random.randint(low=0, high=np.iinfo(np.int32).max, size=n_configurations),
            env=env,
            policy=eval_policy,
        )
        logger.record_tabular_misc_stat('FinalReward', [path["rewards"][-1] for path in eval_paths])
        logger.record_tabular_misc_stat('TotalReward', [np.sum(path["rewards"]) for path in eval_paths])
        env.log_diagnostics(eval_paths)
        logger.dump_tabular()

        # compute optimal actions
        logger.log("Annotating new trajectories")
        fetch_utils.annotate_paths(eval_paths)
        trainer.add_paths(eval_paths)


for batch_size in [512, 1024, 2048, 4096]:

    for hidden_sizes in [(64, 64), (128, 128)]:#(256, 256), (512, 512), (1024, 1024), (1024, 1024, 1024)]:
        # run_cirrascale(
        run_ec2(
        # run_local_docker(
            run_task,
            exp_name="fetch-relative-general-dagger-1-4",
            variant=dict(
                batch_size=batch_size,
                hidden_sizes=hidden_sizes
            ),
            seed=0,
        )
