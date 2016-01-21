require_relative '../utils'

params = {
  mdp: {
    _name: "mujoco_1_22.ant_mdp",
  },
  normalize_mdp: true,
  policy: {
    _name: "mean_std_nn_policy",
    hidden_sizes: [32, 32],
  },
  baseline: {
    _name: "parallel.linear_feature_baseline",
  },
  exp_name: "ant",
  algo: {
    _name: "vpg",
    whole_paths: true,
    batch_size: 1000,
  },
  n_parallel: 1,
  snapshot_mode: "last",
  seed: 1,
}
command = to_profile_command(params)
puts command
system(command)
