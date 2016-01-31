require_relative './utils'

(1..10).each do |seed|
  (0.1..1.0).step(0.1).each do |quantile|
    quantile = quantile.round(2)
    params = {
      mdp: {
        _name: "mountain_car_mdp",
      },
      normalize_mdp: nil,
      policy: {
        _name: "tabular_policy",
        hidden_layers: [],
      },
      vf: {
        _name: "mujoco_value_function",
      },
      exp_name: "ppo_mountain_car_quantile_#{quantile}_seed_#{seed}",
      algo: {
        _name: "ppo",
        binary_search_penalty: false,
        whole_paths: true,
        max_opt_itr: 100,
        quantile: quantile,
        batch_size: 50000,
        max_path_length: 500,
        n_itr: 50,
      },
      n_parallel: 4,
      snapshot_mode: "none",
      seed: seed,
    }

    command = to_command(params)
    puts command
    system(command)
  end
end
