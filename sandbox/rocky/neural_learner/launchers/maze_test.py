from rllab.misc.instrument import stub, run_experiment_lite
# from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rocky.tf.policies.categorical_gru_policy import CategoricalGRUPolicy
from sandbox.rocky.neural_learner.envs.random_maze_env import RandomMazeEnv
from sandbox.rocky.neural_learner.envs.multi_env import MultiEnv
from sandbox.rocky.neural_learner.envs.partial_obs_maze_env import PartialObsMazeEnv
from sandbox.rocky.neural_learner.envs.choice_env import ChoiceEnv
from rllab.envs.grid_world_env import GridWorldEnv
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp

stub(globals())

from rllab.misc.instrument import VariantGenerator

vg = VariantGenerator()
vg.add("seed", [11, 21, 31, 41, 51])
# vg.add("algo", ["trpo"])#, "ppo"])

for v in vg.variants():
    episode_env = PartialObsMazeEnv(RandomMazeEnv(n_row=7, n_col=7))#, seed_pool=[1, 5]))#, 5]))#, seed_pool=[1, 5])#,
    # 5])#,
    #  2])
    # episode_env = ChoiceEnv([
    #     GridWorldEnv(desc=[
    #         "S...",
    #         "....",
    #         "....",
    #         "...G"
    #     ]),
    #     GridWorldEnv(desc=[
    #         "S...",
    #         "....",
    #         "....",
    #         "G..."
    #     ]),
    #     GridWorldEnv(desc=[
    #         "S..G",
    #         "....",
    #         "....",
    #         "...."
    #     ]),
    # ])
    # while True:
    #     episode_env.render()
    env = TfEnv(MultiEnv(
        wrapped_env=episode_env, n_episodes=2, episode_horizon=100, discount=0.99,
    ))
    policy = CategoricalGRUPolicy(name="policy", env_spec=env.spec)
    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        max_path_length=200,
        batch_size=10000,
        discount=0.99,
        n_itr=1000,
        sampler_args=dict(n_envs=10),
        optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="rlrl-maze",
        mode="lab_kube",
        n_parallel=0,
        seed=v["seed"],
        variant=v,
        snapshot_mode="last",
    )
    # sys.exit()
