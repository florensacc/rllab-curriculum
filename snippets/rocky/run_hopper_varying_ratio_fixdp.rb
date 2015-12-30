require_relative './utils'

log_min_ratio = Math.log(1)
log_max_ratio = Math.log(1e1)

n_ratios = 50

log_ratios = n_ratios.times.map { rand * (log_max_ratio - log_min_ratio) + log_min_ratio }
ratios = log_ratios.map{|x| Math.exp(x) }.select{|r| r >= 1.9 and r <= 2.2}


[20, 40, 60, 100].each do |tau|
    [3, 10, 50, 100].each do |k|
        [100, 1e4, 1e6, 1e7, 1e8].each do |fix_rate|
            (1..10).each do |seed|
                ratios.each do |ratio|
                    forward_coeff = 1.0
                    alive_coeff = ratio * forward_coeff
                    exp_name = "fixed_fixdp_ppo_hopper_seed_#{seed}_ratio_#{ratio}_tau_#{tau}_k_#{k}_fix_rate_#{fix_rate}"
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
                        exp_name: exp_name,
                        algo: {
                            _name: "ppo",
                            binary_search_penalty: false,
                            whole_paths: true,
                            batch_size: (30000 ).to_i,
                            max_path_length: 500,
                            n_itr: 100,
                            step_size: 0.01,
                            tau: tau,
                            k: k,
                            fix_rate: fix_rate,
                            # count_fix_part: true,
                        },
                        n_parallel: 8,
                        snapshot_mode: "last",
                        seed: seed,
                    }
                    command = to_command(params)
                    dockerified = """docker run \
  -v ~/.bash_history:/root/.bash_history \
  -v /slave/theano_cache_docker:/root/.theano \
  -v /slave/theanorc:/root/.theanorc \
  -v ~/.vim:/root/.vim \
  -v /slave/gitconfig:/root/.gitconfig \
  -v ~/.vimrc:/root/.vimrc \
  -v /slave/dockerfiles/ssh:/root/.ssh \
  -v /slave/jupyter:/root/.jupyter \
  -v /home/ubuntu/data:/root/workspace/data \
  -v /slave/workspace:/root/workspace \
  -v `pwd`/rllab:/root/workspace/rllab \
  --device /dev/nvidia0:/dev/nvidia0 \
  --device /dev/nvidiactl:/dev/nvidiactl \
  --device /dev/nvidia-uvm:/dev/nvidia-uvm \
  dementrock/starcluster #{command}"""
                    # puts dockerified
                    # system(dockerified)
                    fname = "#{exp_name}.sh"
                    f = File.open(fname, "w")
                    f.puts dockerified
                    f.close
                    system("chmod +x " + fname)
                    system("qsub -V -b n -l mem_free=8G,h_vmem=14G -r y -cwd " + fname) 
                end
            end
        end
    end
end
