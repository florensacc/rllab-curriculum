import pyprind

from rllab.misc import logger
from rllab.sampler.utils import rollout
from sandbox.rocky.new_analogy.exp_utils import run_local_docker, run_cirrascale, run_ec2, run_local

"""
Dagger but with multiple initial positions, for two blocks
"""


def run_task(vv):
    from sandbox.rocky.new_analogy import fetch_utils
    from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
    import tensorflow as tf
    import numpy as np

    horizon = 1000
    n_configurations = 100
    batch_size = vv["batch_size"]  # 512#4096 #512
    n_updates_per_epoch = 10000
    max_n_paths = 10000
    max_n_samples = max_n_paths * horizon

    # n_updates_per_epoch = 10#100
    # n_configurations = 5#10

    max_demo_ratio = vv["max_demo_ratio"]
    min_demo_ratio = vv["min_demo_ratio"]
    ratio_decay_epochs = vv["ratio_decay_epochs"]
    clip_loss = vv["clip_loss"]

    env = fetch_utils.fetch_env(horizon=horizon, height=3)
    demo_policy = fetch_utils.fetch_prescribed_policy(env=env)

    policy = fetch_utils.FetchWrapperPolicy(
        env_spec=env.spec,
        wrapped_policy=GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=vv["hidden_sizes"],
            hidden_nonlinearity=tf.nn.tanh,
            name="policy"
        )
    )

    logger.log("Generating initial demonstrations")
    paths = fetch_utils.new_policy_paths(
        seeds=np.random.randint(low=0, high=np.iinfo(np.int32).max, size=n_configurations),
        env=env,
        policy=demo_policy,
        noise_levels=[0., 0.001, 0.01, 0.1]
    )
    logger.log("Generated")

    trainer = fetch_utils.bc_trainer(env=env, policy=policy, max_n_samples=max_n_samples, clip_square_loss=clip_loss)
    trainer.add_paths(paths)

    logger.log("success rate: {}".format(np.mean([env.wrapped_env._is_success(p) for p in paths])))

    eval_policy = fetch_utils.DeterministicPolicy(env_spec=env.spec, wrapped_policy=policy)
    mixture_policy = fetch_utils.MixturePolicy([demo_policy, eval_policy], ratios=[0., 1.])

    for epoch_idx in trainer.train_loop(batch_size=batch_size, n_updates_per_epoch=n_updates_per_epoch):
        logger.log("Sampling on-policy trajectory")
        mixture_policy.ratios = np.asarray([0., 1.])
        eval_paths = fetch_utils.new_policy_paths(
            seeds=np.random.randint(low=0, high=np.iinfo(np.int32).max, size=n_configurations),
            env=env,
            policy=mixture_policy,
        )
        mixture_ratio = max_demo_ratio - (max_demo_ratio - min_demo_ratio) * min(1., epoch_idx / ratio_decay_epochs)
        logger.log("Sampling mixture policy trajectory")
        mixture_policy.ratios = np.asarray([mixture_ratio, 1 - mixture_ratio])
        addn_paths = fetch_utils.new_policy_paths(
            seeds=np.random.randint(low=0, high=np.iinfo(np.int32).max, size=n_configurations),
            env=env,
            policy=mixture_policy,
        )
        logger.record_tabular('MixtureRatio', mixture_ratio)
        with logger.tabular_prefix("Eval"):
            try:
                env.log_diagnostics(eval_paths)
                logger.record_tabular_misc_stat('FinalReward', [path["rewards"][-1] for path in eval_paths])
                logger.record_tabular_misc_stat('TotalReward', [np.sum(path["rewards"]) for path in eval_paths])
                logger.record_tabular_misc_stat('PathLength', [len(path["rewards"]) for path in eval_paths])
            except Exception as e:
                import ipdb; ipdb.set_trace()
        with logger.tabular_prefix("Mixture"):
            env.log_diagnostics(addn_paths)
            logger.record_tabular_misc_stat('FinalReward', [path["rewards"][-1] for path in addn_paths])
            logger.record_tabular_misc_stat('TotalReward', [np.sum(path["rewards"]) for path in addn_paths])
            logger.record_tabular_misc_stat('PathLength', [len(path["rewards"]) for path in addn_paths])
        logger.dump_tabular()

        # compute optimal actions
        logger.log("Annotating new trajectories")
        fetch_utils.annotate_paths(eval_paths + addn_paths)
        trainer.add_paths(eval_paths + addn_paths)


for batch_size in [4096]:

    for hidden_sizes in [(256, 256), (512, 512)]:

        for clip_loss in [None, 1.]:

            for mixture_configs in [
                dict(max_demo_ratio=0.9, min_demo_ratio=0.1, ratio_decay_epochs=100),
                dict(max_demo_ratio=0.9, min_demo_ratio=0.5, ratio_decay_epochs=100),
                dict(max_demo_ratio=0.9, min_demo_ratio=0.01, ratio_decay_epochs=100),
                dict(max_demo_ratio=0.99, min_demo_ratio=0.01, ratio_decay_epochs=100),
            ]:
                # run_local_docker(
                # run_cirrascale(
                run_ec2(
                    run_task,
                    exp_name="fetch-relative-general-dagger-3-3",
                    variant=dict(
                        batch_size=batch_size,
                        hidden_sizes=hidden_sizes,
                        clip_loss=clip_loss,
                        **mixture_configs
                    ),
                    seed=0,
                    n_parallel=0,
                )
                # sys.exit()
