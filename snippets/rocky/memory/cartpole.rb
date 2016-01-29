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
    #binary_search_penalty: false,
    whole_paths: true,
    batch_size: 10000,
    max_path_length: 100,
    n_itr: 500,
    whole_paths: true,
    step_size: 0.1,
    #max_opt_itr: 20,
  },
  n_parallel: 4,
  snapshot_mode: "none",
  seed: 1,
}
command = to_command(params)
puts command
system(command)

