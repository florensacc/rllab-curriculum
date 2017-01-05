# TODOs
 - TODO's in code
 - make a better interface than main.py. Maybe have a seperate flag parser file

# Works
critic LR: 1.079e-5
actor LR: 1.039e-5
discount: 0.949
reward scale: 1.166

50 sweeps
789 minutes
BATCH_SIZE = 32
N_EPOCHS = 200
EPOCH_LENGTH = 1000
N_EVAL_SAMPLES = 1000
DISCOUNT = 0.99
CRITIC_LEARNING_RATE = 1e-3
ACTOR_LEARNING_RATE = 1e-3
SOFT_TARGET_TAU = 1e-3
REPLAY_POOL_SIZE = 1000000
NUM_EXPERIMENTS_PER_ENV = 1
NUM_HYPERPARAMETER_CONFIGS = 50


sweep7
 - critic LR
 - tau
 - reward scale
    - lower seems to make Q loss more stable
Only one that kinda learned:
 - reward scale = 351.397
 - critic LR = 1.09e-6
 - tau = 0.00647

big2
    hp.FixedParam("discount", 0.99),
    hp.LogFloatParam("actor_learning_rate", 1e-5, 1e-3),
    hp.LogFloatParam("critic_learning_rate", 1e-7, 1e-2),
    hp.LinearFloatParam("tau", 1e-6, 1e-3),
    hp.LogFloatParam("reward_scale", 1e0, 1e3),

rllab-private: half-cheetah-sweep-11-12
Best:
    - reward-scale = 2.37, 1.64, 4.9
    - tau= 0.00389, 0.0144, 0.03455


11-13-post-sweep shane
BATCH_SIZE = 32
N_EPOCHS = 2000
EPOCH_LENGTH = 1000
EVAL_SAMPLES = 1000
DISCOUNT = 0.99
CRITIC_LEARNING_RATE = 1e-3
ACTOR_LEARNING_RATE = 1e-4
SOFT_TARGET_TAU = 0.003819
REPLAY_POOL_SIZE = 1000000
MIN_POOL_SIZE = 10000
SCALE_REWARD = 2.379


# References
reference-ddpg-f0b31b80ecfe06248285b3a2242758ca03ec3dd3/ means that 'f0b31b80ecfe06248285b3a2242758ca03ec3dd3/' is the commit number for this experiments

## reference-ddpg-f0b31b80ecfe06248285b3a2242758ca03ec3dd3/
 - Cartpole
 - Max score oscillates between 1000 and 2500
 - Regularization was wrong

## reference-ddpg-cheetah-87c7cbd1147f9ab2aa83780664dddf499448f737
