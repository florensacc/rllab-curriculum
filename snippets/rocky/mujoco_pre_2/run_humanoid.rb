require_relative '../utils'

params = {
  mdp: {
    _name: "john_mjc2.IcmlHumanoidMDP",
  },
  normalize_mdp: true,
  policy: {
    _name: "mean_std_nn_policy",
    hidden_sizes: [32, 32],
  },
  baseline: {
    _name: "par_nn_baseline",
    #_name: "linear_feature_baseline",
    hidden_sizes: [],#300, 300],
    #max_opt_itr: 500,
  },
  exp_name: "humanoid_mujoco_pre_2",
  algo: {
    _name: "par_ppo",
    whole_paths: true,
    batch_size: 5000,
    max_path_length: 500,
    n_itr: 1,
    step_size: 0.01,
    binary_search_penalty: false,
  },
  n_parallel: 4,
  snapshot_mode: "last",
  seed: 1,
}
command = to_command(params)
puts command
system(command)
