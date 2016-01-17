require_relative './utils'

params = {
  mdp: {
    _name: "nips.cheetah_mdp",
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
    binary_search_penalty: false,
    whole_paths: true,
    batch_size: 10000,
    max_path_length: 100,
    n_itr: 40,
  },
  n_parallel: 4,
  snapshot_mode: "none",
  seed: 1,
}
command = to_command(params)
puts command
system(command)
