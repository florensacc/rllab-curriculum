require_relative './utils'

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
    _name: "par_nn_baseline",
    hidden_sizes: [],
  },
  exp_name: "swimmer",
  algo: {
    _name: "par_ppo",
    whole_paths: true,
    batch_size: 1000,
    max_path_length: 500,
    n_itr: 2,
    step_size: 0.01,
    binary_search_penalty: false,
  },
  n_parallel: 2,
  snapshot_mode: "last",
  seed: 1,
}
command = to_command(params)
puts command
system(command)
