from rllab.misc.instrument import run_experiment_lite
from sandbox.rocky.s3.resource_manager import tmp_file_name, resource_manager
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.envs.base import TfEnv
import tensorflow as tf
import cloudpickle


def run_task(*_):
    env = TfEnv(normalize(CartpoleEnv()))

    policy = GaussianMLPPolicy(name="policy", env_spec=env.spec)

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        batch_size=10000,
        max_path_length=100,
        n_itr=100,
        env=env,
        policy=policy,
        baseline=baseline,
    )

    with tf.Session() as sess:
        algo.train(sess)
        file_name = tmp_file_name(file_ext="pkl")
        with open(file_name, "wb") as f:
            cloudpickle.dump(dict(env=env, policy=policy, baseline=baseline), f, protocol=3)
        resource_name = "pretrained_models/cartpole.pkl"
        resource_manager.register_file(resource_name, file_name=file_name)


run_experiment_lite(
    run_task,
    use_cloudpickle=True,
    exp_prefix="trpo-cartpole",
    mode="local",
)
