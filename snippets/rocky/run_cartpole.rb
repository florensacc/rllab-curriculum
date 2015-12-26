require_relative './utils'

quantile = 1
seed = 1

params = {
  mdp: {
    _name: "box2d.cartpole_mdp",
  },
  normalize_mdp: true,
  policy: {
    _name: "mean_std_nn_policy",
    hidden_sizes: [],
  },
  vf: {
    _name: "mujoco_value_function",
  },
  # exp_name: "ppo_box2d_cartpole_quantile_#{quantile}_seed_#{seed}",
  algo: {
    _name: "ppo",
    binary_search_penalty: false,
    whole_paths: true,
    quantile: quantile,
    batch_size: 1000,
    max_path_length: 100,
    n_itr: 40,
  },
  n_parallel: 1,
  snapshot_mode: "none",
  seed: seed,
}
command = to_command(params)
puts command
system(command)

