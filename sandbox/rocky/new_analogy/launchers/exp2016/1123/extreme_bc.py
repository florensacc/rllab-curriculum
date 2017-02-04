"""
Extreme behavior cloning. Imitate on a single trajectory
"""

import os

import numpy as np

from rllab import config
from rllab.misc import logger
from rllab.misc.instrument import run_experiment_lite
from sandbox.rocky.tf.envs.base import TfEnv

# MODE = "local"


MODE = "local_docker"
# MODE = launch_cirrascale("pascal")


def run_task(*_):
    import tensorflow as tf
    from sandbox.rocky.new_analogy.envs.gpr_env import GprEnv

    with tf.Session() as sess:
        # resource_name = "irl/claw-bc-pretrained-v1.pkl"

        logger.log("Loading data...")
        data = np.load("/shared-data/claw-500-data.npz")
        exp_x = data["exp_x"]
        exp_u = data["exp_u"]
        exp_rewards = data["exp_rewards"]
        logger.log("Loaded")
        paths = []

        for xs, us, rewards in zip(exp_x, exp_u, exp_rewards):
            if rewards[-1] > 4.5:
                paths.append(dict(observations=xs, actions=us, rewards=rewards))

        path = paths[0]

        env = TfEnv(GprEnv("TF2", xinits=[path["observations"][0]]))

        env.reset()
        obs = []
        for a in path["actions"]:
            obs.append(env.step(a)[0])
        obs = np.asarray(obs)
        import ipdb;
        ipdb.set_trace()


        pass

        # file_name = resource_manager.get_file(resource_name=resource_name)

        # policy = joblib.load(file_name)["policy"]
        #
        #
        # discriminator = GCLCostLearnerFixing(
        #     env_spec=env.spec,
        #     demo_paths=paths,
        #     policy=policy,
        #     n_epochs=2,
        #     hidden_sizes=(256, 256),
        #     hidden_nonlinearity=tf.nn.relu,
        #     learning_rate=v["disc_learning_rate"],
        #     demo_mixture_ratio=v["demo_mixture_ratio"],
        #     include_actions=v["disc_include_actions"],
        #     cost_form=v["disc_cost_form"],
        # )
        #
        # baseline = GaussianMLPBaseline(
        #     env_spec=env.spec,
        #     regressor_args=dict(
        #         use_trust_region=True,
        #         hidden_sizes=(256, 256, 256),
        #         hidden_nonlinearity=tf.nn.relu,
        #         optimizer=ConjugateGradientOptimizer(),
        #         step_size=0.1,
        #     ),
        # )
        #
        # sess.run(tf.assign(policy._l_std_param.param, [np.log(vv["init_std"])] * env.action_dim))
        #
        # algo = TRPO(
        #     env=env,
        #     policy=policy,
        #     baseline=baseline,
        #     batch_size=50000,
        #     max_path_length=100,
        #     n_itr=5000,
        #     discount=0.995,
        #     gae_lambda=0.97,
        #     parallel_vec_env=True,
        #     n_vectorized_envs=100,
        #     sample_processor_cls=GAILSampleProcessor,
        #     sample_processor_args=dict(discriminator=discriminator),
        # )
        #
        # algo.train(sess=sess)


# variants = VG().variants()

# print("#Experiments:", len(variants))

if MODE == "local":
    env = dict(PYTHONPATH=":".join([
        config.PROJECT_PATH,
        os.path.join(config.PROJECT_PATH, "conopt_root"),
    ]))
else:
    env = dict(PYTHONPATH="/root/code/rllab:/root/code/rllab/conopt_root")

if MODE in ["local_docker"]:
    env["CUDA_VISIBLE_DEVICES"] = "1"

# for v in variants:
run_experiment_lite(
    run_task,
    use_cloudpickle=True,
    exp_prefix="extreme-bc",
    mode=MODE,
    use_gpu=True,
    snapshot_mode="last",
    sync_all_data_node_to_s3=False,
    n_parallel=0,
    env=env,
    docker_image="quay.io/openai/rocky-rllab3-conopt-gpu-pascal",
    docker_args=" -v /home/rocky/conopt-shared-data:/shared-data",
    variant=dict(),  # v,
    seed=0,  # v["seed"],
)
# sys.exit()
