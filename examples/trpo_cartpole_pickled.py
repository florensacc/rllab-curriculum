from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
# from sandbox.young_clgan.lib.envs.base import FixedStateGenerator
# from sandbox.young_clgan.lib.envs.maze.point_maze_env import PointMazeEnv
from sandbox.young_clgan.envs.mjc_key.pr2_key_env import PR2KeyEnv


def run_task(*_):
    #env = normalize(CartpoleEnv())
    #env = normalize(PointMazeEnv())

    snapshot_mode = 'gap'
    snapshot_gap = 10

    env = normalize(PR2KeyEnv())

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(64, 64)
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
<<<<<<< Updated upstream
        batch_size=4000,
        max_path_length=100,
=======
        batch_size=50000, #4000,
        max_path_length=300,
>>>>>>> Stashed changes
        n_itr=1000,
        discount=0.99,
        step_size=0.01,
        snapshot_mode=snapshot_mode,
        snapshot_gap=snapshot_gap,
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        plot=True,
    )
    algo.train()


run_experiment_lite(
    run_task,
    # Number of parallel workers for sampling
    n_parallel=1,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=1,
    variant=dict(),
    plot=True,
)
