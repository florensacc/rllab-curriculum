require_relative './utils'

seed = 1

params = {
  mdp: {
    # _name: "box2d.hopper_mdp",
    _name: "box2d.cartpole_mdp",
  },
  # normalize_mdp: nil,
  policy: {
    _name: "mean_std_nn_policy",
    hidden_sizes: [32, 32],
  },
  baseline: {
    _name: "linear_feature_baseline",
  },
  exp_name: "ppo_box2d_hopper",
  algo: {
    # _name: "ppo",
    _name: "npg",
    whole_paths: true,
    batch_size: 5000,
    max_path_length: 500,
    n_itr: 500,
    step_size: 0.01,
    update_method: 'sgd',
    learning_rate: 0.9,
    plot: true,
  },
  n_parallel: 4,
  snapshot_mode: "last",
  plot: true,
  seed: seed,
}
command = to_command(params)
puts command
system(command)
