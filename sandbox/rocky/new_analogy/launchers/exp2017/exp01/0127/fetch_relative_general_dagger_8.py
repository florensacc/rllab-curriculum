from rllab.misc import logger
from sandbox.rocky.new_analogy.exp_utils import run_ec2
from sandbox.rocky.new_analogy.tf.policies.auto_mlp_policy import AutoMLPPolicy

"""
Dagger but with multiple initial positions, for two blocks
"""


def run_task(vv):
    from sandbox.rocky.new_analogy import fetch_utils
    import tensorflow as tf
    import numpy as np

    n_boxes = vv["n_boxes"]  # 3

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

    per_stage_horizon = 500
    n_configurations = 100
    batch_size = vv["batch_size"]  # 512#4096 #512
    n_updates_per_epoch = 10000
    max_n_paths = 10000
    max_n_samples = max_n_paths * horizon

    # n_updates_per_epoch = 10  # 100
    # n_configurations = 5  # 100#5#10
    # horizon = 500
    # batch_size = 128

    max_demo_ratio = vv["max_demo_ratio"]
    min_demo_ratio = vv["min_demo_ratio"]
    ratio_decay_epochs = vv["ratio_decay_epochs"]
    clip_loss = vv["clip_loss"]


    disc_intervals = np.asarray([
        [-0.3, -0.1, -0.03, -0.01, -0.003, -0.001, -0.0003, 0, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
        [-0.3, -0.1, -0.03, -0.01, -0.003, -0.001, -0.0003, 0, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
        [-0.3, -0.1, -0.03, -0.01, -0.003, -0.001, -0.0003, 0, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
    ])

    # disc_intervals = [
    #     [-0.05, -0.01, -0.001, 0, 0.001, 0.01, 0.05],
    #     [-0.05, -0.01, -0.001, 0, 0.001, 0.01, 0.05],
    #     [-0.05, -0.01, -0.001, 0, 0.001, 0.01, 0.05],
    # ]

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
    _paths = fetch_utils.new_policy_paths(
        seeds=np.random.randint(low=0, high=np.iinfo(np.int32).max, size=n_configurations),
        env=env,
        policy=demo_policy,
        noise_levels=[0., 1e-3, 1e-2]
    )
    logger.log("Generated")

    # filter out failed paths
    stage_init_pts = []
    paths = []
    for p in _paths:
        pts = fetch_utils.find_stageinit_points(env, p)
        if len(pts) == n_boxes:
            stage_init_pts.append(pts)
            paths.append(p)

    stage_xinits = []
    for pts, path in zip(stage_init_pts, paths):
        stage_xinits.extend(path["env_infos"]["x"][pts[:-1]])

    # use stage as the first index
    stage_xinits = np.asarray(stage_xinits)

    trainer = fetch_utils.bc_trainer(env=env, policy=nn_policy, max_n_samples=max_n_samples, clip_square_loss=clip_loss)
    trainer.add_paths(paths)

    logger.log("success rate: {}".format(np.mean([env.wrapped_env._is_success(p) for p in _paths])))

    eval_policy = fetch_utils.DeterministicPolicy(env_spec=env.spec, wrapped_policy=nn_policy)
    mixture_policy = fetch_utils.MixturePolicy([demo_policy, eval_policy], ratios=[0., 1.])

    for epoch_idx in trainer.train_loop(batch_size=batch_size, n_updates_per_epoch=n_updates_per_epoch):
        logger.log("Sampling on-policy trajectory")
        mixture_policy.ratios = np.asarray([0., 1.])
        eval_paths = fetch_utils.new_policy_paths(
            seeds=np.random.randint(low=0, high=np.iinfo(np.int32).max, size=n_configurations),
            env=env,
            policy=mixture_policy,
            horizon=horizon,
        )

        # in this kind of training, only execute the policy until a single stage has been completed
        logger.log("Sampling on-policy stagewise trajectory")
        stagewise_paths = fetch_utils.new_policy_paths(
            seeds=np.random.randint(low=0, high=np.iinfo(np.int32).max, size=n_configurations),
            env=env,
            policy=mixture_policy,
            horizon=per_stage_horizon,
            xinits=stage_xinits,
            stagewise=True,
        )

        logger.log("Sampling stagewise mixture trajectory")
        mixture_ratio = max_demo_ratio - (max_demo_ratio - min_demo_ratio) * min(1., epoch_idx / ratio_decay_epochs)
        mixture_policy.ratios = np.asarray([mixture_ratio, 1 - mixture_ratio])
        addn_paths = fetch_utils.new_policy_paths(
            seeds=np.random.randint(low=0, high=np.iinfo(np.int32).max, size=n_configurations),
            env=env,
            policy=mixture_policy,
            horizon=per_stage_horizon,
            xinits=stage_xinits,
            stagewise=True
        )

        logger.record_tabular('MixtureRatio', mixture_ratio)
        with logger.tabular_prefix("Eval|"):
            env.log_diagnostics(eval_paths)
            logger.record_tabular_misc_stat('FinalReward', [path["rewards"][-1] for path in eval_paths])
            logger.record_tabular_misc_stat('TotalReward', [np.sum(path["rewards"]) for path in eval_paths])
            logger.record_tabular_misc_stat('PathLength', [len(path["rewards"]) for path in eval_paths])
        with logger.tabular_prefix("MixtureStagewise|"):
            env.log_diagnostics(addn_paths)
            logger.record_tabular_misc_stat('FinalReward', [path["rewards"][-1] for path in addn_paths])
            logger.record_tabular_misc_stat('TotalReward', [np.sum(path["rewards"]) for path in addn_paths])
            logger.record_tabular_misc_stat('PathLength', [len(path["rewards"]) for path in addn_paths])
        with logger.tabular_prefix("OnPolicyStagewise|"):
            env.log_diagnostics(stagewise_paths)
            logger.record_tabular_misc_stat('FinalReward', [path["rewards"][-1] for path in stagewise_paths])
            logger.record_tabular_misc_stat('TotalReward', [np.sum(path["rewards"]) for path in stagewise_paths])
            logger.record_tabular_misc_stat('PathLength', [len(path["rewards"]) for path in stagewise_paths])
        logger.dump_tabular()

        all_paths = eval_paths + addn_paths + stagewise_paths

        # compute optimal actions
        logger.log("Annotating new trajectories")
        fetch_utils.annotate_paths(all_paths)
        for p in all_paths:
            p["actions"] = demo_policy.discretize_actions(p["actions"])
        trainer.add_paths(all_paths)


for batch_size in [128, 256, 512, 1024]:

    for hidden_sizes in [(256, 256)]:  # , (512, 512)]:

        for clip_loss in [1.]:  # None, 1.]:

            for n_boxes in [2, 3, 4, 5]:

                for mixture_configs in [
                    dict(max_demo_ratio=0.99, min_demo_ratio=0.1, ratio_decay_epochs=100),
                    # dict(max_demo_ratio=0.9, min_demo_ratio=0.5, ratio_decay_epochs=100),
                    # dict(max_demo_ratio=0.9, min_demo_ratio=0.01, ratio_decay_epochs=100),
                    # dict(max_demo_ratio=0.99, min_demo_ratio=0.01, ratio_decay_epochs=100),
                ]:
                    for seed in [100]:#, 200, 300]:
                        # run_local_docker(
                        # run_local(
                        # run_cirrascale(
                        run_ec2(
                            run_task,
                            exp_name="fetch-relative-general-dagger-8",
                            variant=dict(
                                batch_size=batch_size,
                                hidden_sizes=hidden_sizes,
                                clip_loss=clip_loss,
                                n_boxes=n_boxes,
                                seed=seed,
                                **mixture_configs
                            ),
                            seed=seed,
                            n_parallel=0,
                            # use_gpu=False,
                        )
                        # exit()
