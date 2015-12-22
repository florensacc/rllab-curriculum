require_relative './utils'

log_min_ratio = Math.log(1e-1)
log_max_ratio = Math.log(1e1)

n_ratios = 50

log_ratios = n_ratios.times.map { rand * (log_max_ratio - log_min_ratio) + log_min_ratio }
ratios = log_ratios.map{|x| Math.exp(x) }

(1..10).each do |seed|
  ratios.each do |ratio|
    forward_coeff = 1.0
    alive_coeff = ratio * forward_coeff
    # TODO configure log_dir to under /home
    params = {
      mdp: {
        _name: "mujoco.hopper_mdp",
        forward_coeff: forward_coeff,
        alive_coeff: alive_coeff,
      },
      policy: {
        _name: "mean_std_nn_policy",
        hidden_layers: [32, 32],
      },
      vf: {
        _name: "mujoco_value_function",
      },
      exp_name: "ppo_hopper_seed_#{seed}_ratio_#{ratio}",
      algo: {
        _name: "ppo",
        binary_search_penalty: false,
        whole_paths: true,
        batch_size: 50000,
        max_path_length: 500,
        n_itr: 200,
        step_size: 0.01,
      },
      n_parallel: 8,
      snapshot_mode: "last",
      seed: seed,
    }
    command = to_command(params)
    puts command
    system(command)
  end
end
