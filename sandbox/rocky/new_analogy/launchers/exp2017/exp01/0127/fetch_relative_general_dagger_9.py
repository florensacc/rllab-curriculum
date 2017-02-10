from rllab.misc import logger
from sandbox.rocky.new_analogy.exp_utils import run_ec2
from sandbox.rocky.new_analogy.tf.policies.auto_mlp_policy import AutoMLPPolicy

"""
Try without dagger, new obs space
"""


def run_task(vv):
    from sandbox.rocky.new_analogy import fetch_utils
    import tensorflow as tf
    import numpy as np

    n_boxes = vv["n_boxes"]

    if n_boxes == 3:
        horizon = 1000
    elif n_boxes == 2:
        horizon = 500
    elif n_boxes == 4:
        horizon = 1500
    elif n_boxes == 5:
        horizon = 2000
    else:
        raise NotImplementedError

    n_configurations = 2000
    batch_size = vv["batch_size"]
    n_updates_per_epoch = 10000
    max_n_paths = n_configurations
    max_n_samples = max_n_paths * horizon

    # n_updates_per_epoch = 10
    # n_configurations = 5
    # horizon = 500
    # batch_size = 128

    disc_intervals = np.asarray([
        [-0.3, -0.1, -0.03, -0.01, -0.003, -0.001, -0.0003, 0, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
        [-0.3, -0.1, -0.03, -0.01, -0.003, -0.001, -0.0003, 0, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
        [-0.3, -0.1, -0.03, -0.01, -0.003, -0.001, -0.0003, 0, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
    ])

    env = fetch_utils.fetch_env(horizon=horizon, height=n_boxes)
    demo_policy = fetch_utils.fetch_discretized_prescribed_policy(env=env, disc_intervals=disc_intervals)

    nn_policy = fetch_utils.DiscretizedFetchWrapperPolicy(
        wrapped_policy=AutoMLPPolicy(
            env_spec=fetch_utils.discretized_env_spec(env.spec, disc_intervals),
            hidden_sizes=vv["hidden_sizes"],
            hidden_nonlinearity=tf.nn.tanh,
            name="policy"
        ),
        disc_intervals=disc_intervals
    )

    logger.log("Generating initial demonstrations")
    paths = fetch_utils.new_policy_paths(
        seeds=np.random.randint(low=0, high=np.iinfo(np.int32).max, size=n_configurations),
        env=env,
        policy=demo_policy,
        noise_levels=noise_levels,
    )
    logger.log("Generated")

    trainer = fetch_utils.bc_trainer(env=env, policy=nn_policy, max_n_samples=max_n_samples)
    trainer.add_paths(paths)

    logger.log("success rate: {}".format(np.mean([env.wrapped_env._is_success(p) for p in paths])))

    eval_policy = fetch_utils.DeterministicPolicy(env_spec=env.spec, wrapped_policy=nn_policy)

    for _ in trainer.train_loop(batch_size=batch_size, n_updates_per_epoch=n_updates_per_epoch):
        logger.log("Sampling on-policy trajectory")
        eval_paths = fetch_utils.new_policy_paths(
            seeds=np.arange(100),
            env=env,
            policy=eval_policy,
            horizon=horizon,
        )

        env.log_diagnostics(eval_paths)
        logger.record_tabular_misc_stat('FinalReward', [path["rewards"][-1] for path in eval_paths])
        logger.record_tabular_misc_stat('TotalReward', [np.sum(path["rewards"]) for path in eval_paths])
        logger.record_tabular_misc_stat('PathLength', [len(path["rewards"]) for path in eval_paths])
        logger.dump_tabular()


for batch_size in [512]:#, 1024, 4096]:

    for hidden_sizes in [(256, 256)]:

        for n_boxes in [2, 3, 4, 5]:

            for noise_levels in [[0., 1e-3, 1e-2], [0.], [0., 1e-3]]:

                for seed in [100]:
                    # run_cirrascale(
                    # run_local_docker(
                    run_ec2(
                        run_task,
                        exp_name="fetch-relative-general-dagger-9",
                        variant=dict(
                            batch_size=batch_size,
                            hidden_sizes=hidden_sizes,
                            n_boxes=n_boxes,
                            noise_levels=noise_levels,
                            seed=seed,
                            # **mixture_configs
                        ),
                        seed=seed,
                        n_parallel=0,
                    )
                    # exit()
