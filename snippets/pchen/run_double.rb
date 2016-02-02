require_relative '../rocky/utils'

quantile = 1
seed = 1

params = {
  mdp: {
    # _name: "box2d.double_pendulum_mdp",
    # _name: "box2d.car_parking_mdp",
    _name: "box2d.cartpole_mdp",
    # trig_angle: false,
    # frame_skip: 2,
  },
  normalize_mdp: true,
  obs_noise: 0.05,
  action_delay: 0,
  policy: {
    _name: "mean_std_nn_policy",
    # hidden_layers: [],
  },
  baseline: {
    _name: "linear_feature_baseline",
  },
  exp_name: "ppo_double_skip2_pendulum_seed_#{seed}",
  algo: {
    _name: "ppo",
    binary_search_penalty: false,
    whole_paths: true,
    # # quantile: quantile,

    # _name: "reps",
    batch_size: 5000,
    max_path_length: 100,
    n_itr: 500,
    plot: true,
    # step_size: 0.1,

  },
  n_parallel: 1,
  # snapshot_mode: "none",
  seed: seed,
  plot: true,
}
command = to_command(params)
puts command
system(command)

