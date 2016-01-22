require_relative '../utils'

params = {
  mdp: {
    _name: "box2d.double_pendulum_mdp",
    position_only: true,
  },
  normalize_mdp: true,
  policy: {
    _name: "mean_std_nn_policy",
  },
  baseline: {
    _name: "linear_feature_baseline",
  },
  exp_name: "double_part_obs",
  algo: {
    _name: "ppo",
    binary_search_penalty: false,
    whole_paths: true,
    # quantile: quantile,
    batch_size: 10000,
    max_path_length: 100,
    n_itr: 500,
    plot: true,
    step_size: 0.1,
  },
  n_parallel: 4,
  # snapshot_mode: "none",
  seed: 1,
  plot: true,
}
command = to_command(params)
puts command
system(command)
