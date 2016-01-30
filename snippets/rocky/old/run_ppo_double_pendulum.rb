require_relative '../rocky/utils'

quantile = 1
seed = 1

params = {
  mdp: {
    _name: "box2d.double_pendulum_mdp",
    # trig_angle: false,
    # frame_skip: 2,
  },
  # normalize_mdp: nil,
  policy: {
    _name: "mean_std_nn_policy",
    # hidden_layers: [],
  },
  baseline: {
    _name: "linear_feature_baseline",
  },
  exp_name: "ppo_double_pendulum",
  algo: {
    _name: "ppo",
    binary_search_penalty: true,
    bs_kl_tolerance: 0.001,
    whole_paths: true,
    batch_size: 10000,
    max_path_length: 100,
    n_itr: 500,
    step_size: 0.01,
  },
  n_parallel: 4,
  # snapshot_mode: "none",
  seed: seed,
}
command = to_command(params)
puts command
system(command)

