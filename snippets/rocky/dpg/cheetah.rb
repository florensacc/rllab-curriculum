require_relative '../utils'

# seed = 1

params = {
  mdp: {
    _name: "mujoco_1_22.half_cheetah_mdp",
    #action_noise: 0.01,
  },
  normalize_mdp: true,
  qf: {
    _name: "continuous_nn_q_function",
    hidden_sizes: [100, 100],#32, 32],
    normalize: false,
    #output_nl: 'lasa
    #output_nl: 'lasagne.nonlinearities.tanh',
    bn: true,#true,
  },
  policy: {
    _name: "mean_nn_policy",
    hidden_sizes: [400, 300],#32, 32],#32,32],
    output_nl: 'lasagne.nonlinearities.tanh',
    bn: true,#true,
  },
  algo: {
    _name: "dpg",
    batch_size: 64,
    n_epochs: 1000,
    epoch_length: 1000,
    min_pool_size: 64,#10000,
    replay_pool_size: 1000000,
    discount: 0.99,
    qf_weight_decay: 1e-2,#0,#1e-2,#0,#1e-3,
    qf_learning_rate: 1e-3,
    max_path_length: 100,
    eval_samples: 10000,
    eval_whole_paths: true,
    soft_target: true,
    #hard_target_interval: 1000,
    soft_target_tau: 1e-3,
    policy_learning_rate: 1e-4,
  },
  es: {
    _name: "ou_strategy",
    theta: 0.15,
    sigma: 0.2,
    #_name: "gaussian_strategy",
    #max_sigma: 0.5,#1.0,
    #min_sigma: 0.5,#0.1,
    #sigma_decay_range: 1000000,
  },
  n_parallel: 4,
  # seed: seed,
}
command = to_command(params)
puts command
system(command)

