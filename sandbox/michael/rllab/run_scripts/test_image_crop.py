from examples.fixed_crop_env import FixedCropEnv

from rllab.algos.trpo import TRPO
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

stub(globals())

env = normalize(FixedCropEnv(max_action= 6, radius = 10, max_init_distance= 6))
policy = GaussianMLPPolicy(
    hidden_sizes= (256, 256),
    env_spec=env.spec,
)
baseline = GaussianMLPBaseline(env_spec=env.spec)
algo = TRPO(
    env=env,
    policy=policy,
    max_path_length=3,
    n_itr=10000, # iterations
    batch_size= 10000, # trajectories = batch size / max_path_length
    baseline=baseline,
    # plot = True
)
# algo.train()
run_experiment_lite(
    algo.train(),
    n_parallel=4,
    snapshot_mode="last",
    exp_prefix="fixedcrop10",
    use_gpu=True
    # plot = True
)

#11: decrease how much network can move
#12 make network bigger, correctness condition fixed
#14: continue training
#15: increase path length, network size 
#16: increase batch size


#basic #6 bonus for done
#basic #14 use gpu to train, bonus for finishing 128x128 architecture
# 15 n_itr should be multiple of max path length
#16 switch back to 200x200
#17 fix batch size
#19 seems to work pretty well (around 60% of the time)
#20 random initialization, max path 12
#21 test hypothesis that longer path is bad
#24 max path length 10

#fixedcrop
#reward of 5 for finishing
#1. max_path_length = 8, batch_size = 10000
#2  max_path_length = 8, batch_size = 10000, max_action = 10
#3  max_path_length = 6, batch_size = 10000, max_action = 10

#4 make box a lot smaller, max_path_length = 6
#5 make box a lot smaller, max_path_length = 10

#6 decrease threshold for correctness max_path_length = 6
#7 max_path_length = 4

#8 max_path_length = 8
#9 max_path_length = 4
#10 trained for 5000 iterations, policy does well
