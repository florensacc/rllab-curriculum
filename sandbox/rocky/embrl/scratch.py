from rllab.envs.gym_env import GymEnv
from rllab.misc import logger
from sandbox.rocky.embrl.core import EnsembleModelEnv, EMBRL
from sandbox.rocky.new_analogy.exp_utils import run_local, run_ec2
from sandbox.rocky.tf.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp
from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer
import numpy as np

from sandbox.rocky.tf.policies.categorical_rnn_policy import CategoricalRNNPolicy
from sandbox.rocky.tf.regressors.auto_mlp_regressor import AutoMLPRegressor
from sandbox.rocky.tf.spaces import Product, Box, Discrete


def run_task(vv):
    env = TfEnv(GymEnv("CartPole-v0", record_video=False, record_log=False, force_reset=True))

    policy = CategoricalRNNPolicy(
        name="policy",
        env_spec=env.spec,
        hidden_dim=32,
    )

    n_models = vv["n_models"]#20
    max_n_updates = vv["max_n_updates"]#1000

    model_output_space = Product(
        env.observation_space,
        Box(low=-np.inf, high=np.inf, shape=(1,)),
        Discrete(2),
    )

    models = []

    for idx in range(n_models):
        logger.log("Constructing models")
        model = AutoMLPRegressor(
            name="model_{}".format(idx),
            input_shape=(env.observation_space.flat_dim + env.action_space.flat_dim,),
            output_space=model_output_space,
            optimizer=FirstOrderOptimizer(max_epochs=max_n_updates, max_updates=max_n_updates),
            use_trust_region=False,
            hidden_sizes=(32, 32),
            separate_networks=True,
        )
        models.append(model)

    ensemble_env = EnsembleModelEnv(env_spec=env.spec, models=models, model_output_space=model_output_space)

    algo = EMBRL(
        env=env,
        policy=policy,
        ensemble_env=ensemble_env,
        n_itr=1000,
        trpo_args=dict(
            n_itr=100,
            baseline=LinearFeatureBaseline(env_spec=env.spec),
            batch_size=10000,
            discount=0.99,
            horizon=env.horizon,
            optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5)),
        )
    )
    algo.train()


for n_models in [2, 5, 10, 20]:
    for max_n_updates in [100, 1000]:
        for seed in [100, 200, 300]:
            run_ec2(
                run_task,
                exp_name="embrl-1",
                variant=dict(n_models=n_models, max_n_updates=max_n_updates, seed=seed),
                seed=seed,
                instance_type="c4.2xlarge",
            )
