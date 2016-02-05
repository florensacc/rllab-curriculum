require_relative '../rocky/utils'

quantile = 1
seed = 1

params = {
  mdp: {
    # _name: "box2d.cartpole_swingup_mdp",
    # _name: "box2d.double_pendulum_mdp",
    _name: "mujoco_1_22.half_cheetah_mdp",
  },
  normalize_mdp: true,
  policy: {
    _name: "uniform_control_policy",
  },
  baseline: {
    _name: "zero_baseline",
  },
  exp_name: "hehe#{seed}",
  algo: {
    _name: "nop",
    batch_size: 5000,

    whole_paths: true,
    max_path_length: 100,
    n_itr: 5000,
    plot: true,
  },
  n_parallel: 1,
  snapshot_mode: "last",
  seed: seed,
  plot: true,
}
command = to_command(params)
puts command
system(command)

