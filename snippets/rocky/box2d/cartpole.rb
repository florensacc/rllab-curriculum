require_relative '../utils'

params = {
  mdp: {
    _name: "box2d.cartpole_mdp",
  },
  normalize_mdp: true,
  policy: {
    _name: "mean_std_nn_policy",
    hidden_sizes: [],
  },
  baseline: {
    _name: "linear_feature_baseline",
  },
  #exp_name: "radian_ppo_box2d_cartpole_swingup_quantile_#{quantile}_seed_#{seed}",
  algo: {
    _name: "ppo",
    binary_search_penalty: false,
    whole_paths: true,
    batch_size: 10000,
    max_path_length: 100,
    n_itr: 500,
    step_size: 0.1,
  },
  n_parallel: 4,
  # snapshot_mode: "none",
  #seed: seed,
}
command = to_command(params)
puts command
system(command)

