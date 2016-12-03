"""Test different rl algorithms."""
import argparse

import tensorflow as tf

from algos.ddpg import DDPG as MyDDPG
from algos.naf import NAF
from algos.noop_algo import NoOpAlgo
from misc import hyperparameter as hp
from policies.nn_policy import FeedForwardPolicy
from qfunctions.nn_qfunction import FeedForwardCritic
from qfunctions.quadratic_naf_qfunction import QuadraticNAF
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.gym_env import GymEnv
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.normalized_env import normalize
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.exploration_strategies.gaussian_strategy import GaussianStrategy
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.uniform_control_policy import UniformControlPolicy
from sandbox.rocky.tf.algos.ddpg import DDPG as ShaneDDPG
from sandbox.rocky.tf.policies.deterministic_mlp_policy import \
    DeterministicMLPPolicy
from sandbox.rocky.tf.q_functions.continuous_mlp_q_function import \
    ContinuousMLPQFunction

BATCH_SIZE = 64
N_EPOCHS = 10
EPOCH_LENGTH = 50
EVAL_SAMPLES = 10
DISCOUNT = 0.99
CRITIC_LEARNING_RATE = 1e-3
ACTOR_LEARNING_RATE = 1e-4
SOFT_TARGET_TAU = 0.01
REPLAY_POOL_SIZE = 1000000
MIN_POOL_SIZE = 10
SCALE_REWARD = 1.0
Q_WEIGHT_DECAY = 0.0
MAX_PATH_LENGTH = 1000

SWEEP_N_EPOCHS = 50
SWEEP_MIN_POOL_SIZE = BATCH_SIZE

NUM_SEEDS_PER_CONFIG = 2
NUM_HYPERPARAMETER_CONFIGS = 50


def get_env_settings(args):
    env_name = args.env
    if env_name == 'cart':
        env = CartpoleEnv()
        name = "Cartpole"
    elif env_name == 'cheetah':
        env = HalfCheetahEnv()
        name = "HalfCheetah"
    elif env_name == 'point':
        env = normalize(GymEnv("Pointmass-v1", record_video=False))
        name = "Pointmass"
    else:
        raise Exception("Unknown env: {0}".format(env_name))

    return dict(
        env=env,
        name=name,
    )


def get_algo_settings(args):
    algo_name = args.algo
    if algo_name == 'ddpg':
        sweeper = hp.HyperparameterSweeper([
            hp.LogFloatParam("soft_target_tau", 0.005, 0.1),
            hp.LogFloatParam("scale_reward", 10.0, 0.01),
            hp.LogFloatParam("Q_weight_decay", 1e-7, 1e-1),
        ])
        params = get_my_ddpg_params()
        test_function = test_my_ddpg
    elif algo_name == 'naf':
        sweeper = hp.HyperparameterSweeper([
            hp.LogFloatParam("qf_learning_rate", 1e-4, 1e-2),
            hp.LogFloatParam("scale_reward", 10.0, 0.01),
        ])
        params = get_my_naf_params()
        test_function = test_my_naf
    elif algo_name == 'random':
        sweeper = hp.HyperparameterSweeper()
        params = {}
        test_function = test_random_ddpg

    else:
        raise Exception("Algo name not recognized: " + algo_name)

    params['render'] = args.render
    return {
        'sweeper': sweeper,
        'algo_params': params,
        'test_function': test_function,
    }


def get_ddpg_params():
    return dict(
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        epoch_length=EPOCH_LENGTH,
        eval_samples=EVAL_SAMPLES,
        discount=DISCOUNT,
        policy_learning_rate=ACTOR_LEARNING_RATE,
        qf_learning_rate=CRITIC_LEARNING_RATE,
        soft_target_tau=SOFT_TARGET_TAU,
        replay_pool_size=REPLAY_POOL_SIZE,
        min_pool_size=MIN_POOL_SIZE,
        scale_reward=SCALE_REWARD,
        max_path_length=MAX_PATH_LENGTH,
    )


def get_my_ddpg_params():
    params = get_ddpg_params()
    params['Q_weight_decay'] = Q_WEIGHT_DECAY
    return params


def get_my_naf_params():
    return dict(
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        epoch_length=EPOCH_LENGTH,
        eval_samples=EVAL_SAMPLES,
        discount=DISCOUNT,
        qf_learning_rate=CRITIC_LEARNING_RATE,
        soft_target_tau=SOFT_TARGET_TAU,
        replay_pool_size=REPLAY_POOL_SIZE,
        min_pool_size=MIN_POOL_SIZE,
        scale_reward=SCALE_REWARD,
        max_path_length=MAX_PATH_LENGTH,
        Q_weight_decay=Q_WEIGHT_DECAY
    )


def test_my_ddpg(env, exp_prefix, env_name, seed=1, **ddpg_params):
    es = OUStrategy(env_spec=env.spec)
    qf_params = dict(
        embedded_hidden_sizes=(100,),
        observation_hidden_sizes=(100,),
        # hidden_W_init=util.xavier_uniform_initializer,
        # hidden_b_init=tf.zeros_initializer,
        # output_W_init=util.xavier_uniform_initializer,
        # output_b_init=tf.zeros_initializer,
        hidden_nonlinearity=tf.nn.relu,
    )
    policy_params = dict(
        observation_hidden_sizes=(100, 100),
        # hidden_W_init=util.xavier_uniform_initializer,
        # hidden_b_init=tf.zeros_initializer,
        # output_W_init=util.xavier_uniform_initializer,
        # output_b_init=tf.zeros_initializer,
        hidden_nonlinearity=tf.nn.relu,
        output_nonlinearity=tf.nn.tanh,
    )
    qf = FeedForwardCritic(
        "critic",
        env.observation_space.flat_dim,
        env.action_space.flat_dim,
        **qf_params
    )
    policy = FeedForwardPolicy(
        "actor",
        env.observation_space.flat_dim,
        env.action_space.flat_dim,
        **policy_params
    )
    algorithm = MyDDPG(
        env,
        es,
        policy,
        qf,
        **ddpg_params
    )
    variant = ddpg_params
    variant['Version'] = 'Mine'
    variant['Environment'] = env_name
    for qf_key, qf_value in qf_params.items():
        variant['qf_' + qf_key] = str(qf_value)
    for policy_key, policy_value in policy_params.items():
        variant['policy_' + policy_key] = str(policy_value)
    run_experiment(algorithm, exp_prefix, seed, variant)


def test_my_naf(env, exp_prefix, env_name, seed=1, **naf_params):
    es = GaussianStrategy(env)
    qf = QuadraticNAF(
        "qf",
        env.spec,
    )
    algorithm = NAF(
        env,
        es,
        qf,
        **naf_params
    )
    variant = naf_params
    variant['Version'] = 'Mine'
    variant['Environment'] = env_name
    variant['Algo'] = 'NAF'
    run_experiment(algorithm, exp_prefix, seed, variant)


def test_shane_ddpg(env, exp_prefix, env_name, seed=1, **new_ddpg_params):
    ddpg_params = dict(get_ddpg_params(), **new_ddpg_params)
    es = GaussianStrategy(env.spec)

    policy_params = dict(
        hidden_sizes=(100, 100),
        hidden_nonlinearity=tf.nn.relu,
        output_nonlinearity=tf.nn.tanh,
    )
    qf_params = dict(
        hidden_sizes=(100, 100)
    )
    policy = DeterministicMLPPolicy(
        name="init_policy",
        env_spec=env.spec,
        **policy_params
    )
    qf = ContinuousMLPQFunction(
        name="qf",
        env_spec=env.spec,
        **qf_params
    )

    algorithm = ShaneDDPG(
        env,
        policy,
        qf,
        es,
        **ddpg_params
    )

    variant = ddpg_params
    variant['Version'] = 'Shane'
    variant['Environment'] = env_name
    for qf_key, qf_value in qf_params.items():
        variant['qf_' + qf_key] = str(qf_value)
    for policy_key, policy_value in policy_params.items():
        variant['policy_' + policy_key] = str(policy_value)

    run_experiment(algorithm, exp_prefix, seed, variant=variant)


def test_random_ddpg(env, exp_prefix, env_name, seed=1, **algo_params):
    es = OUStrategy(env)
    policy = UniformControlPolicy(env_spec=env.spec)
    algorithm = NoOpAlgo(
        env,
        policy,
        es,
        **algo_params)
    variant = {'Version': 'Random', 'Environment': env_name}

    run_experiment(algorithm, exp_prefix, seed, variant=variant)


def run_experiment(algorithm, exp_prefix, seed, variant):
    variant['seed'] = str(seed)
    print("variant=")
    print(variant)
    run_experiment_lite(
        algorithm.train(),
        n_parallel=1,
        snapshot_mode="last",
        exp_prefix=exp_prefix,
        variant=variant,
        seed=seed,
    )


def sweep(exp_prefix, env_settings, algo_settings):
    sweeper = algo_settings['sweeper']
    test_function = algo_settings['test_function']
    default_params = algo_settings['algo_params']
    env = env_settings['env']
    env_name = env_settings['name']
    for i in range(NUM_HYPERPARAMETER_CONFIGS):
        for seed in range(NUM_SEEDS_PER_CONFIG):
            params = dict(default_params,
                          **sweeper.generate_random_hyperparameters())
            params['n_epochs'] = SWEEP_N_EPOCHS
            params['min_pool_size'] = SWEEP_MIN_POOL_SIZE
            test_function(env, exp_prefix, env_name, seed=seed + 1,
                          **params)


def main():
    env_choices = ['cheetah', 'cart', 'point']
    algo_choices = ['ddpg', 'naf', 'shane-ddpg', 'random']
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action='store_true',
                        help="Run benchmarks.")
    parser.add_argument("--sweep", action='store_true',
                        help="Sweep hyperparameters for my DDPG.")
    parser.add_argument("--render", action='store_true',
                        help="Render the environment.")
    parser.add_argument("--env", default='cart',
                        help="Test algo on 'cart' or 'cheetah'.",
                        choices=env_choices)
    parser.add_argument("--name", default='default',
                        help='Experiment prefix')
    parser.add_argument("--algo", default='ddpg',
                        help='Algo',
                        choices=algo_choices)
    parser.add_argument("--seed", default=1,
                        type=int,
                        help='Seed')
    args = parser.parse_args()

    stub(globals())

    algo_settings = get_algo_settings(args)
    env_settings = get_env_settings(args)
    if args.sweep:
        sweep(args.name, env_settings, algo_settings)
    else:
        test_function = algo_settings['test_function']
        algo_params = algo_settings['algo_params']
        env = env_settings['env']
        env_name = env_settings['name']
        test_function(env, args.name, env_name, seed=args.seed, **algo_params)


if __name__ == "__main__":
    main()
