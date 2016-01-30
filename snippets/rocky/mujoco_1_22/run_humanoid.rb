require_relative '../utils'

params = {
  mdp: {
    _name: "john_mjc2.IcmlHumanoidMDP",#mujoco_pre_2.humanoid_amputated_mdp",
  },
  normalize_mdp: true,
  policy: {
    _name: "mean_std_nn_policy",
    hidden_sizes: [100, 50, 32],
  },
  baseline: {
    #_name: "nn_baseline",
    _name: "parallel.linear_feature_baseline",
    #hidden_sizes: [],#300, 300],
    #max_opt_itr: 500,
  },
  exp_name: "humanoid",
  algo: {
    _name: "parallel.trpo",
    whole_paths: true,
    batch_size: 50000,
    max_path_length: 500,
    n_itr: 10000,
    step_size: 1,
    plot: true,
  },
  n_parallel: 4,
  snapshot_mode: "last",
  seed: 1,
  plot: true,
}
command = to_command(params)
puts command
system(command)
