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
    binary_search_penalty: true,
    bs_kl_tolerance: 1e-3,
    whole_paths: true,
    batch_size: 50000,
    max_path_length: 500,
    n_itr: 500,
    step_size: 0.01,
    increase_penalty_factor: 5,
    decrease_penalty_factor: 0.1,
    # plot: true,
  },
  n_parallel: 4,
  snapshot_mode: "last",
  # plot: true,
  seed: seed,
}
command = to_command(params)
puts command
system(command)
