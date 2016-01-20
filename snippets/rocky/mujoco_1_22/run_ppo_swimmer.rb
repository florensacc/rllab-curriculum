require_relative '../utils'

params = {
  mdp: {
    _name: "mujoco_1_22.swimmer_mdp",
  },
  normalize_mdp: true,
  policy: {
    _name: "mean_std_nn_policy",
    hidden_sizes: [32, 32],
  },
  baseline: {
    _name: "linear_feature_baseline",
  },
  exp_name: "swimmer_mujoco_1_22_ppo",
  algo: {
    _name: "ppo",
    whole_paths: true,
    batch_size: 4000,
    max_path_length: 500,
    n_itr: 500,
    binary_search_penalty: false,
    # bs_kl_tolerance: 0.001,
    step_size: 0.1,
    plot: true,
  },
  plot: true,
  n_parallel: 4,
  snapshot_mode: "last",
  seed: 1,
}
command = to_command(params)
puts command
system(command)
