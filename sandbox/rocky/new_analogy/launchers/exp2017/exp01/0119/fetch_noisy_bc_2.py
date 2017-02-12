from sandbox.rocky.new_analogy import exp_utils

"""
Behavior cloning on fetch using noisy trajectories
"""


def run_task():
    import tensorflow as tf
    from sandbox.rocky.new_analogy.tf.algos import Trainer
    from sandbox.rocky.s3.resource_manager import resource_manager
    from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
    from sandbox.rocky.new_analogy import fetch_utils
    from rllab.misc import logger
    import pickle

    logger.log("Loading data")
    resource_names = [
        "tower_fetch_ab_annotated",
        "tower_fetch_on_policy_round_1_ab_annotated"
    ]
    paths = []
    for resource_name in resource_names:
        file_name = resource_manager.get_file(resource_name)
        with open(file_name, "rb") as f:
            paths.extend(pickle.load(f))

    for p in paths:
        p["actions"] = p["taught_actions"]
        del p["taught_actions"]
    logger.log("Loaded")

    with tf.Session() as sess:

        env = fetch_utils.fetch_env()

        policy = fetch_utils.FetchWrapperPolicy(
            env_spec=env.spec,
            wrapped_policy=GaussianMLPPolicy(
                env_spec=env.spec,
                hidden_sizes=(512, 512, 512, 512),
                hidden_nonlinearity=tf.nn.relu,
                name="policy"
            )
        )

        algo = Trainer(
            env=env,
            policy=policy,
            paths=paths,
            n_epochs=5000,
            n_passes_per_epoch=1,
            evaluate_performance=True,
            train_ratio=0.95,
            max_path_length=1000,
            n_eval_trajs=10,
            eval_batch_size=10000,
            n_eval_envs=10,
            batch_size=1024,
            n_slices=20,
            learn_std=False,
        )

        algo.train(sess=sess)


exp_utils.run_local_docker(
    run_task,
    exp_name="fetch-noisy-bc-2",
)
