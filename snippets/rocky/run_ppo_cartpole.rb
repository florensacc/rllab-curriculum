require_relative './utils'

params = {
  mdp: {
    _name: "box2d.cartpole_mdp",
    position_only: true,
  },
  normalize_mdp: true,
  policy: {
    _name: "mean_std_rnn_policy",
    hidden_sizes: [],
  },
  baseline: {
    _name: "zero_baseline",
  },
  algo: {
    _name: "ppo",
    binary_search_penalty: false,
    whole_paths: true,
    batch_size: 1000,
    max_path_length: 100,
    n_itr: 40,
  },
  n_parallel: 1,
  snapshot_mode: "none",
  seed: 1,
}
command = to_command(params)
puts command
system(command)
