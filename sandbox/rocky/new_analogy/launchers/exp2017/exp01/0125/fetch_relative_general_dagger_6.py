import pyprind

from rllab.misc import logger
from rllab.sampler.utils import rollout
from sandbox.rocky.new_analogy.exp_utils import run_local_docker, run_cirrascale, run_ec2, run_local

"""
Dagger, only train for the second block
"""


def run_task(vv):
    from sandbox.rocky.new_analogy import fetch_utils
    from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
    import tensorflow as tf
    import numpy as np

    horizon = 1000
    per_stage_horizon = 500
    n_configurations = 100
    batch_size = vv["batch_size"]  # 512#4096 #512
    n_updates_per_epoch = 10000
    max_n_paths = 10000
    max_n_samples = max_n_paths * horizon

    n_boxes = vv["n_boxes"]  # 3

    # n_updates_per_epoch = 10  # 100
    # n_configurations = 5  # 100#5#10

    max_demo_ratio = vv["max_demo_ratio"]
    min_demo_ratio = vv["min_demo_ratio"]
    ratio_decay_epochs = vv["ratio_decay_epochs"]
    clip_loss = vv["clip_loss"]

    env = fetch_utils.fetch_env(horizon=horizon, height=n_boxes)
    demo_policy = fetch_utils.fetch_prescribed_policy(env=env)

    nn_policy = fetch_utils.FetchWrapperPolicy(
        env_spec=env.spec,
        wrapped_policy=GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=vv["hidden_sizes"],
            hidden_nonlinearity=tf.nn.tanh,
            name="policy"
        )
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

    import ipdb; ipdb.set_trace()

    # use stage as the first index
    stage_xinits = np.asarray(stage_xinits)  # .transpose((1, 0, 2))
    # flatten_stage_xinits = stage_xinits.reshape((n_boxes-1 * len(paths), stage_xinits.shape[-1]))

    trainer = fetch_utils.bc_trainer(env=env, policy=nn_policy, max_n_samples=max_n_samples, clip_square_loss=clip_loss)
    trainer.add_paths(paths)

    logger.log("success rate: {}".format(np.mean([env.wrapped_env._is_success(p) for p in paths])))

    eval_policy = fetch_utils.DeterministicPolicy(env_spec=env.spec, wrapped_policy=nn_policy)
    mixture_policy = fetch_utils.MixturePolicy([demo_policy, eval_policy], ratios=[0., 1.])

    for epoch_idx in trainer.train_loop(batch_size=batch_size, n_updates_per_epoch=n_updates_per_epoch):
        # logger.log("Sampling on-policy trajectory")
        mixture_policy.ratios = np.asarray([0., 1.])
        # eval_paths = fetch_utils.new_policy_paths(
        #     seeds=np.random.randint(low=0, high=np.iinfo(np.int32).max, size=n_configurations),
        #     env=env,
        #     policy=mixture_policy,
        #     horizon=horizon,
        # )

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
        # with logger.tabular_prefix("Eval|"):
        #     env.log_diagnostics(eval_paths)
        #     logger.record_tabular_misc_stat('FinalReward', [path["rewards"][-1] for path in eval_paths])
        #     logger.record_tabular_misc_stat('TotalReward', [np.sum(path["rewards"]) for path in eval_paths])
        #     logger.record_tabular_misc_stat('PathLength', [len(path["rewards"]) for path in eval_paths])
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

        all_paths = addn_paths + stagewise_paths

        # compute optimal actions
        logger.log("Annotating new trajectories")
        fetch_utils.annotate_paths(all_paths)
        trainer.add_paths(all_paths)


for batch_size in [4096]:

    for hidden_sizes in [(256, 256)]:  # , (512, 512)]:

        for clip_loss in [1.]:  # None, 1.]:

            for n_boxes in [3]:#, 2]:  # 2, 3]:

                for mixture_configs in [
                    dict(max_demo_ratio=0.99, min_demo_ratio=0.1, ratio_decay_epochs=100),
                    # dict(max_demo_ratio=0.9, min_demo_ratio=0.5, ratio_decay_epochs=100),
                    # dict(max_demo_ratio=0.9, min_demo_ratio=0.01, ratio_decay_epochs=100),
                    # dict(max_demo_ratio=0.99, min_demo_ratio=0.01, ratio_decay_epochs=100),
                ]:
                    for seed in [100, 200, 300]:
                        # run_local_docker(
                        # run_cirrascale(
                        run_local(#)
                            run_task,
                            exp_name="fetch-relative-general-dagger-6",
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
                            # use_gpu=True,#False,
                        )
                        exit()
                        # sys.exit()
