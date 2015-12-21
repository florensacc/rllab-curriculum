require_relative '../rocky/utils'

quantile = 1
seed = 1

params = {
  mdp: {
    _name: "box2d.cartpole_swingup_mdp",
  },
  normalize_mdp: nil,
  policy: {
    _name: "mean_std_nn_policy",
    hidden_layers: [],
  },
  vf: {
    _name: "mujoco_value_function",
  },
  exp_name: "cossin_ppo_box2d_cartpole_swingup_quantile_#{quantile}_seed_#{seed}",
  algo: {
    _name: "ppo",
    binary_search_penalty: false,
    whole_paths: true,
    quantile: quantile,
    batch_size: 10000,
    max_path_length: 100,
    n_itr: 500,
  },
  n_parallel: 4,
  # snapshot_mode: "none",
  seed: seed,
}
command = to_command(params)
puts command
system(command)

