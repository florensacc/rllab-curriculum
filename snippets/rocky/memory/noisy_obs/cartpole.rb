require_relative '../../utils'

params = {
  mdp: {
    _name: "box2d.cartpole_mdp",
    #position_only: true,
  },
  normalize_mdp: true,
  obs_noise: 1,
  policy: {
    _name: "mean_std_rnn_policy",
  },
  baseline: {
    _name: "linear_feature_baseline",
  },
  algo: {
    _name: "recurrent.rppo",
    batch_size: 10000,
    whole_paths: true,
    max_path_length: 100,
    n_itr: 500,
    step_size: 0.01,
    n_slices: 10,
  },
  n_parallel: 4,
  snapshot_mode: "none",
  seed: 1,
}
command = to_command(params)
puts command
system(command)
