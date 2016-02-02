require_relative '../utils'

params = {
  mdp: {
    _name: "box2d.cartpole_mdp",
    position_only: true,
  },
  normalize_mdp: true,
  policy: {
    _name: "mean_std_rnn_policy",
  },
  baseline: {
    _name: "linear_feature_baseline",
  },
  algo: {
    _name: "recurrent.rtrpo",
    batch_size: 50000,
    whole_paths: true,
    max_path_length: 500,
    n_itr: 500,
    step_size: 0.01,
  },
  n_parallel: 4,
  snapshot_mode: "none",
  seed: 1,
}
command = to_command(params)
puts command
system(command)
