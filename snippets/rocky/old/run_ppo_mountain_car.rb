require_relative './utils'

params = {
  mdp: {
    _name: "box2d.mountain_car_mdp",
  },
  normalize_mdp: true,
  policy: {
    _name: "mean_std_nn_policy",
    # hidden_sizes: [32, 32],
  },
  baseline: {
    _name: "linear_feature_baseline",
  },
  algo: {
    _name: "ppo",
    binary_search_penalty: false,
    bs_kl_tolerance: 1e-3,
    whole_paths: true,
    batch_size: 10000,
    max_path_length: 100,
    n_itr: 100,
    plot: true,
    step_size: 0.1,
  },
  plot: true,
  n_parallel: 1,
  snapshot_mode: "none",
  seed: 1,
}
command = to_command(params)
puts command
system(command)
