require_relative './utils'

quantile = 1
(3..10).each do |seed|
  step_penalty_coeff = 0.1
  distance_coeff = 0.1
  # (0.1..1.0).step(0.1).each do |step_penalty_coeff|
  #   (0..1.0).step(0.1).each do |distance_coeff|
      step_penalty_coeff = step_penalty_coeff.round(2)
      distance_coeff = distance_coeff.round(2)
      params = {
        mdp: {
          _name: "mountain_car_mdp",
          step_penalty_coeff: step_penalty_coeff,
          distance_coeff: distance_coeff,
        },
        normalize_mdp: nil,
        policy: {
          _name: "tabular_policy",
          hidden_layers: [],
        },
        vf: {
          _name: "mujoco_value_function",
        },
        exp_name: "ppo_mountain_car_step_penalty_coeff_#{step_penalty_coeff}_distance_coeff_#{distance_coeff}_seed_#{seed}",
        algo: {
          _name: "ppo",
          binary_search_penalty: false,
          whole_paths: true,
          max_opt_itr: 20,
          quantile: quantile,
          batch_size: 50000,
          max_path_length: 500,
          n_itr: 50,
        },
        n_parallel: 4,
        snapshot_mode: "last",
        seed: seed,
      }

      command = to_command(params)
      puts command
      system(command)
  #   end
  # end
end
