


from sandbox.rocky.neural_learner.envs.multi_env import MultiEnv
from sandbox.rocky.neural_learner.envs.bimodal_point_env import BimodalPointEnv
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.algos.ppo import PPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.policies.gaussian_gru_policy import GaussianGRUPolicy
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp

from rllab.misc.instrument import stub, run_experiment_lite

stub(globals())

from rllab.misc.instrument import VariantGenerator

vg = VariantGenerator()
vg.add("seed", [11, 21, 31, 41, 51])
vg.add("algo", ["trpo", "ppo"])
# vg.add("algo", ["ppo"])

for v in vg.variants():
    episode_env = BimodalPointEnv()
    trial_env = MultiEnv(wrapped_env=episode_env, n_episodes=5, episode_horizon=20)

    env = TfEnv(normalize(trial_env))

    policy = GaussianGRUPolicy(name="policy", env_spec=env.spec, state_include_action=False)

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    if v["algo"] == "trpo":
        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            max_path_length=100,
            batch_size=10000,
            n_itr=1000,
            sampler_args=dict(n_envs=10),
            optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
        )
    elif v["algo"] == "ppo":
        algo = PPO(
            env=env,
            policy=policy,
            baseline=baseline,
            max_path_length=100,
            batch_size=10000,
            n_itr=1000,
            sampler_args=dict(n_envs=10),
        )
    else:
        raise NotImplementedError

    run_experiment_lite(
        algo.train(),
        exp_prefix="rlrl-bipoint",
        mode="lab_kube",
        n_parallel=4,
        seed=v["seed"],
        variant=v,
        snapshot_mode="last"
    )
