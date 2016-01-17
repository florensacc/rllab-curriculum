require_relative './utils'

params = {
  mdp: {
    _name: "mujoco_pre_2.swimmer_mdp",
  },
  normalize_mdp: true,
  policy: {
    _name: "mean_std_nn_policy",
  },
  baseline: {
    _name: "zero_baseline",
  },
  algo: {
    _name: "ppo",
    binary_search_penalty: true,
    bs_kl_tolerance: 0.001,
    whole_paths: true,
    batch_size: 200000,
    max_path_length: 500,
    step_size: 0.01,
    n_itr: 50,
  },
  n_parallel: 4,
  snapshot_mode: "last",
  seed: 1,
}
command = to_command(params)
puts command
system(command)
