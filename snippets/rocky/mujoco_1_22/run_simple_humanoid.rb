require_relative '../utils'

params = {
  mdp: {
    _name: "mujoco_1_22.simple_humanoid_mdp",
  },
  normalize_mdp: true,
  policy: {
    _name: "mean_std_nn_policy",
    hidden_sizes: [100, 50, 32],
  },
  baseline: {
    _name: "linear_feature_baseline",
  },
  exp_name: "simple_humanoid",
  algo: {
    _name: "trpo",
    whole_paths: true,
    batch_size: 50000,
    max_path_length: 500,
    n_itr: 10000,
    step_size: 1,
    # plot: true,
  },
  n_parallel: 4,
  snapshot_mode: "last",
  seed: 1,
  # plot: true,
}
command = to_command(params)
puts command
system(command)
