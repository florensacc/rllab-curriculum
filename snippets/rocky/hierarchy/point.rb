require_relative '../utils'

# checked

params = {
  mdp: {
    _name: "mujoco_1_22.gather.point_gather_mdp",
    n_apples: 8,
    n_bombs: 8,
    # sensor_span: 2*Math::PI,
    # sensor_range: 6*1.414,
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
    _name: "parallel.trpo",
    whole_paths: true,
    batch_size: 50000,
    max_path_length: 300,
    n_itr: 500,
    step_size: 0.1,
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
