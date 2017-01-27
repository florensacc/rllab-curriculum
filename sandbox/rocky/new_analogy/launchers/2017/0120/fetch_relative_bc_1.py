from sandbox.rocky.new_analogy.exp_utils import run_local_docker, run_cirrascale


def run_task(*_):
    from sandbox.rocky.new_analogy import fetch_utils
    from sandbox.rocky.s3 import resource_manager
    from rllab.misc import logger
    from sandbox.rocky.new_analogy.algos.ff_bc_trainer import Trainer
    from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
    from sandbox.rocky.new_analogy.fetch_utils import FetchWrapperPolicy

    import pickle
    import tensorflow as tf

    logger.log("Loading data...")
    file_name = resource_manager.get_file("fetch_relative/10000_trajs.pkl")
    with open(file_name, "rb") as f:
        paths = pickle.load(f)
    logger.log("Loaded")

    horizon = 300
    n_eval_trajs = 10

    with tf.Session() as sess:
        env = fetch_utils.fetch_env(horizon=horizon)

        paths = list(filter(env.wrapped_env._is_success, paths))
        print("#paths:", len(paths))

        policy = FetchWrapperPolicy(
            env_spec=env.spec,
            wrapped_policy=GaussianMLPPolicy(
                env_spec=env.spec,
                hidden_sizes=(256, 256, 256),
                hidden_nonlinearity=tf.nn.tanh,
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
            max_path_length=horizon,
            n_eval_trajs=n_eval_trajs,
            eval_batch_size=horizon * n_eval_trajs,
            n_eval_envs=n_eval_trajs,
            batch_size=512,
            n_slices=10,
            learn_std=False,
        )

        algo.train(sess=sess)


# run_cirrascale(
run_local_docker(
    run_task,
    exp_name="fetch-relative-bc-1",
    seed=0
)
