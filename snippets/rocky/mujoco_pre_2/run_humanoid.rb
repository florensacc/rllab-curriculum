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
    _name: "nn_baseline",
    #_name: "linear_feature_baseline",
    hidden_sizes: [100, 50, 25],#300, 300],
    max_opt_itr: 5,#500,
  },
  exp_name: "humanoid_mujoco_pre_2",
  algo: {
    _name: "trpo",
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
