require_relative '../rocky/utils'

itrs = 1000
batch_size = 5000
horizon = 100
discount = 0.99
seeds = (1..1).each do |i| i ** 2 * 5 + 23 end

mdps = []
mdps << "box2d.cartpole_mdp"
# mdps << "box2d.mountain_car_mdp"
# mdps << "box2d.cartpole_swingup_mdp"
# mdps << "box2d.double_pendulum_mdp"

algos = []
# npg
# [0.1, 0.01].each do |ss|
#   [1e-2, 1e-1, 1e0].each do |lr|
#     algos << {
#       _name: "npg",
#       step_size: ss,
#       update_method: "adam",
#       learning_rate: lr,
#     }
#   end
# end

algos << {
  _name: "npg",
  step_size: 0.1,
  update_method: "sgd",
  learning_rate: 1,
  use_cg: true,
  cg_iters: 10,
}

inc = 0
seeds.each do |seed|
  mdps.each do |mdp|
    algos.each do |algo|
      exp_name = "npg_test"
      params = {
        mdp: {
          _name: mdp,
        },
        normalize_mdp: true,
        policy: {
          _name: "mean_std_nn_policy",
          hidden_sizes: [32, 32],
        },
        baseline: {
          _name: "linear_feature_baseline",
        },
        exp_name: exp_name,
        algo: {
          whole_paths: true,
          max_path_length: horizon,
          n_itr: itrs,
          discount: discount,
          plot: true,
        }.merge(algo),
        n_parallel: 2,
        snapshot_mode: "last",
        seed: seed,
        plot: true,
      }
      command = to_command(params)
      # puts command
      system(command)
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
  dementrock/starcluster:new #{command}"""
      # puts dockerified
      # system(dockerified)
      # fname = "#{exp_name}.sh"
      # f = File.open(fname, "w")
      # f.puts dockerified
      # f.close
      # system("chmod +x " + fname)
      # system("qsub -V -b n -l mem_free=8G,h_vmem=14G -r y -cwd " + fname)
    end
  end
end
