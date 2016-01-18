require_relative './utils'

seed = 1

params = {
  mdp: {
    _name: "box2d.cartpole_mdp",
  },
  normalize_mdp: true,
  qf: {
    _name: "continuous_nn_q_function",
    hidden_sizes: [32, 32],
    normalize: false,
    bn: true,
  },
  policy: {
    _name: "mean_nn_policy",
    hidden_sizes: [32,32],#100, 100],#32, 32],
    hidden_nl: 'lasagne.nonlinearities.rectify',
    output_nl: 'lasagne.nonlinearities.tanh',
    output_W_init: 'lasagne.init.Uniform(-3e-3, 3e-3)',
    output_b_init: 'lasagne.init.Uniform(-3e-3, 3e-3)',
    bn: true,
  },
  exp_name: "dpg_box2d_cartpole",
  algo: {
    _name: "dpg",
    batch_size: 32,
    n_epochs: 1000,
    epoch_length: 1000,
    min_pool_size: 10000,
    replay_pool_size: 1000000,
    discount: 0.99,
    qf_weight_decay: 1e-2,
    qf_learning_rate: 1e-3,
    max_path_length: 100,
    eval_samples: 1000,
    eval_whole_paths: true,
    soft_target_tau: 0.001,
    policy_learning_rate: 1e-4,#1e-4,#0.0001,
  },
  es: {
    _name: "ou_strategy",#gaussian_strategy",
    #max_sigma: 1,
    #min_sigma: 0.01,
    #sigma_decay_range: 200000,
  },
  #n_parallel: 1,
  #snapshot_mode: "none",
  seed: seed,
}
command = to_command(params)
puts command
system(command)

