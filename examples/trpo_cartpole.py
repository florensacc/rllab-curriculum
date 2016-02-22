import sys
sys.path.append(".")

from rllab.misc.console import run_experiment

params = dict(
    mdp=dict(
        # Specifies that we want to run the MDP under the module path
        # rllab.mdp.box2d.cartpole_mdp
        _name="box2d.cartpole_mdp",
    ),
    # Normalizes the action range for the MDP to lie between -1 and 1
    normalize_mdp=True,
    policy=dict(
        # Specifies that we want to use the policy representation under the
        # module path rllab.policy.mean_std_nn_policy
        _name="mean_std_nn_policy",
        # Specifies that the neural network policy should have two hidden
        # layers, each with 32 hidden units.
        hidden_sizes=[32, 32],
    ),
    baseline=dict(
        # Specifies that we want to use the baseline under the module path
        # rllab.baseline.linear_feature_baseline
        _name="linear_feature_baseline",
    ),
    # The experiment log will be stored in the folder data/trpo_cartpole
    exp_name="trpo_cartpole",
    algo=dict(
        # Specifies that we want to run the algorithm under the module path
        # rllab.algo.trpo
        _name="trpo",
        # Each iteration of the algorithm will rollout roughly 4000 samples
        # under the current stochastic policy
        batch_size=4000,
        # Each iteration might yield slightly more than 4000 samples, to make
        # sure that all trajectories are sampled until termination, or until
        # the horizon is reached
        whole_paths=True,
        # Maximum horizon for each trajectory.
        max_path_length=100,
        # Number of iterations
        n_itr=40,
        # Discount factor
        discount=0.99,
        # KL-divergence step size for the algorithm
        step_size=0.01,
        # Uncomment the line below (and the plot=True below) to visualize the
        # policy performance
        # plot=True,
    ),
    # Number of parallel workers for sampling
    n_parallel=1,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is blank, a random seed
    # will be used
    seed=1,
    # plot=True,
)

run_experiment(params)
