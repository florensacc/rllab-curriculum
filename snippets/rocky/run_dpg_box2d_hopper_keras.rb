require_relative './utils'

seeds = (1..10).to_a

hiddens = [[100, 100]]#, 100]]#32, 32, 32]]#, [100, 100], [32, 32, 32], [100, 100, 100]]
qf_weight_decays = [0, 1e-5, 1e-6, 1e-7]
# policy_lrs = [1e-4, 5e-4, 1e-5]
discounts = [0.99, 0.999, 0.9999, 1]
batch_sizes = [64]#32, 64, 128]
epoch_lengths = [10000]#, 10000, 20000, 40000]
# qf_lrs = [1e-3, 1e-4, 1e-5]


def sample_lr
  log_min_lr = Math.log(1e-6)
  log_max_lr = Math.log(1e-3)
  Math.exp(rand * (log_max_lr - log_min_lr) + log_min_lr)
end


1000.times do
  params = {
    mdp: {
      _name: "box2d.hopper_mdp",
    },
    normalize_mdp: true,
    qf: {
      _name: "continuous_nn_keras_q_function",
      hidden_sizes: hiddens.sample,
    },
    policy: {
      _name: "mean_nn_keras_policy",
      hidden_sizes: hiddens.sample,
      output_nl: 'tanh',
    },
    # exp_name: "dpg_box2d_cartpole_swingup",
    algo: {
      _name: "dpg",
      batch_size: batch_sizes.sample,
      n_epochs: 500,
      epoch_length: epoch_lengths.sample,
      min_pool_size: 50000,
      replay_pool_size: 500000,
      discount: discounts.sample,
      qf_weight_decay: qf_weight_decays.sample,
      max_path_length: 500,
      eval_samples: 500,
      eval_whole_paths: true,
      renormalize_interval: 5000,
      # normalize_qval: false,
      policy_learning_rate: sample_lr,#policy_lrs.sample,
      qf_learning_rate: sample_lr,#policy_lrs.sample,
    },
    es: {
      _name: "ou_strategy",
    },
    n_parallel: 1,
    snapshot_mode: "last",
    seed: seeds.sample,
  }
  # command = to_docker_command(params)
  # system(command)
  # break
  create_task_script(to_docker_command(params), launch: true, prefix: "dpg_hopper")
  # puts command
  # system(command)
end
