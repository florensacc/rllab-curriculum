require_relative '../utils'

# checked

params = {
  mdp: {
    _name: "mujoco_1_22.maze_mdp",
  },
  normalize_mdp: true,
  policy: {
    _name: "mean_std_nn_policy",
    hidden_sizes: [32, 32],
  },
  baseline: {
    _name: "linear_feature_baseline",
  },
  exp_name: "point_gather",
  algo: {
    _name: "ppo",
    whole_paths: true,
    batch_size: 5000,
    max_path_length: 300,
    n_itr: 500,
    step_size: 0.01,
    # plot: true,
  },
  n_parallel: 1,
  snapshot_mode: "last",
  seed: 1,
  # plot: true,
}
command = to_command(params)
puts command
system(command)
