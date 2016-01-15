require_relative './utils'

params = {
  mdp: {
    _name: "box2d.cartpole_mdp",
    position_only: true,
  },
  normalize_mdp: true,
  policy: {
    _name: "mean_std_rnn_policy1",
  },
  baseline: {
    _name: "zero_baseline",
  },
  algo: {
    _name: "rvpg1",
    update_method: "adam",
    learning_rate: 0.01, #e-4,
    batch_size: 10000,
    n_itr: 500,
    max_path_length: 100,
    whole_paths: true,
    # center_adv: false,
    # quantile: quantile,
    # batch_size: 1000,
    # max_path_length: 100,
    # n_itr: 40,
  },
  n_parallel: 4,
  snapshot_mode: "none",
  seed: 1,
}
command = to_command(params)
puts command
system(command)

