require_relative '../utils'

params = {
  mdp: {
    _name: "john_mjc2.IcmlHumanoidMDP",
  },
  normalize_mdp: true,
  policy: {
    _name: "mean_std_nn_policy",
    hidden_sizes: [100, 50, 25],
  },
  baseline: {
    _name: "parallel.linear_feature_baseline",
  },
  exp_name: "humanoid_mujoco_pre_2",
  algo: {
    _name: "parallel.trpo",
    whole_paths: true,
    batch_size: 50000,
    max_path_length: 2000,
    n_itr: 1000,
    step_size: 0.1,
  },
  n_parallel: 4,
  snapshot_mode: "last",
  seed: 1,
}
command = to_command(params)
puts command
system(command)
