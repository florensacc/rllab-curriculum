require_relative './utils'

seed = 1

params = {
  mdp: {
    _name: "box2d.hopper_mdp",
  },
  normalize_mdp: nil,
  policy: {
    _name: "mean_std_nn_policy",
    hidden_layers: [32, 32],
  },
  vf: {
    _name: "mujoco_value_function",
  },
  exp_name: "ppo_box2d_hopper",
  algo: {
    _name: "ppo",
    binary_search_penalty: false,
    whole_paths: true,
    batch_size: 50000,
    max_path_length: 500,
    n_itr: 200,
    step_size: 0.01,
    plot: true,
  },
  n_parallel: 4,
  snapshot_mode: "last",
  plot: true,
  seed: seed,
}
command = to_command(params)
puts command
system(command)
