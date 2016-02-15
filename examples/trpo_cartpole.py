from rllab.misc.console import run_experiment

params = dict(
    mdp="box2d.cartpole_mdp",
    normalize_mdp=True,
    policy=dict(
        _name="mean_std_nn_policy",
        hidden_sizes=[32, 32],
    ),
    baseline=dict(
        _name="linear_feature_baseline",
    ),
    exp_name="trpo_cartpole",
    algo=dict(
        _name="trpo",
        batch_size=4000,
        whole_paths=True,
        max_path_length=100,
        n_itr=40,
        discount=0.99,
        step_size=0.01,
        # plot=True,
    ),
    n_parallel=1,
    snapshot_mode="last",
    seed=1,
    # plot=True,
)

run_experiment(params)
