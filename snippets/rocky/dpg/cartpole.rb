require_relative '../utils'

#seed = 1

params = {
  mdp: {
    _name: "box2d.cartpole_mdp",
  },
  normalize_mdp: true,
  qf: {
    _name: "continuous_nn_q_function",
    hidden_sizes: [32,32],#10,10],#32, 32],
    bn: true,#false,#true,
  },
  policy: {
    _name: "mean_nn_policy",
    hidden_sizes: [32,32],#10,10],#32, 32],
    output_nl: 'lasagne.nonlinearities.tanh',
    bn: true,#false,#true,
  },
  algo: {
    _name: "dpg",
    batch_size: 32,
    n_epochs: 100,
    epoch_length: 1000,
    min_pool_size: 10000,
    replay_pool_size: 100000,
    discount: 0.99,
    qf_weight_decay: 1e-2,
    qf_learning_rate: 1e-3,
    max_path_length: 40,
    eval_samples: 1000,
    eval_whole_paths: true,
    soft_target: true,
    #hard_target_interval: 1000,
    soft_target_tau: 0.001,
    policy_learning_rate: 1e-4,
  },
  es: {
    _name: "ou_strategy",
    theta: 0.15,
    sigma: 0.3,
  },
  n_parallel: 4,
  #seed: seed,
}
command = to_command(params)
puts command
system(command)

