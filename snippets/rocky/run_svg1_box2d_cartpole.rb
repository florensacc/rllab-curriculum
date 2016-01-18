require_relative './utils'

seed = 1

params = {
  mdp: {
    _name: "box2d.cartpole_mdp",
  },
  normalize_mdp: true,
  vf: {
    _name: "nn_value_function",
  },
  policy: {
    _name: "mean_std_nn_policy",
    hidden_sizes: [32, 32],
    output_nl: "lasagne.nonlinearities.tanh",
  },
  exp_name: "dpg_box2d_cartpole",
  algo: {
    _name: "svg1",
    batch_size: 100,
    n_epochs: 100,
    epoch_length: 1000,
    min_pool_size: 10000,
    replay_pool_size: 100000,
    discount: 0.99,
    model_learning_rate: 1e-4,
    max_path_length: 100,
    eval_samples: 10000,
    eval_whole_paths: true,
  },
  model: {
    _name: "mean_nn_model",
  },
  snapshot_mode: "none",
  seed: seed,
}
command = to_command(params)
puts command
system(command)

