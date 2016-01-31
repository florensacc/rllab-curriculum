from rllab.sampler import parallel_sampler


def worker_collect_paths():
    return parallel_sampler.G.paths

#parallel_sampler.config_parallel_sampler(n_parallel=1, base_seed=1)

from rllab.mdp.mujoco_1_22.half_cheetah_mdp import HalfCheetahMDP
from rllab.mdp.normalized_mdp import normalize
from rllab.policy.mean_nn_policy import MeanNNPolicy
from rllab.qf.continuous_nn_q_function import ContinuousNNQFunction
from rllab.misc.ext import compile_function, extract, set_seed
from rllab.es.ou_strategy import OUStrategy
from rllab.algo.util import ReplayPool
from lasagne.updates import adam
import pyprind
import numpy as np
import theano.tensor as TT

set_seed(1)


if __name__ == "__main__":

    class Options(object):

        def __init__(self):
            self.discount = 0.99
            self.epoch_length = 1000
            self.n_epochs = 100
            self.max_path_length = 150
            self.min_pool_size = 1000
            self.max_pool_size = 1e6
            self.batch_size = 64
            self.eval_samples = 1000
            self.soft_target_tau = 0.005

    options = Options()

    mdp = normalize(HalfCheetahMDP())#action_noise=0.01))

    policy = MeanNNPolicy(mdp, hidden_sizes=[400, 300], bn=True, output_nl='lasagne.nonlinearities.tanh')
    target_policy = MeanNNPolicy(mdp, hidden_sizes=[400, 300], bn=True, output_nl='lasagne.nonlinearities.tanh')
    target_policy.set_param_values(policy.get_param_values())

    qf = ContinuousNNQFunction(mdp, hidden_sizes=[100, 100], bn=True)
    target_qf = ContinuousNNQFunction(mdp, hidden_sizes=[100, 100], bn=True)
    target_qf.set_param_values(qf.get_param_values())

    es = OUStrategy(mdp, theta=0.15, sigma=0.3)

    pool = ReplayPool(
        observation_shape=mdp.observation_shape,
        action_dim=mdp.action_dim,
        max_steps=options.max_pool_size,
    )

    parallel_sampler.populate_task(mdp, policy)

    obs_var = TT.matrix('obs')
    action_var = TT.matrix('action')
    next_obs_var = TT.matrix('next_obs')
    reward_var = TT.vector('reward')
    terminal_var = TT.vector('terminal')

    ys_var = reward_var + options.discount * (1 - terminal_var) * \
        target_qf.get_qval_sym(
            next_obs_var,
            target_policy.get_action_sym(
                next_obs_var, deterministic=True
            ),
            deterministic=True
        )

    input_vars = (obs_var, action_var, next_obs_var, reward_var, terminal_var)

    f_ys = compile_function(input_vars, ys_var)

    ys_computed_var = TT.vector('ys')

    qf_loss = TT.mean(TT.square(qf.get_qval_sym(obs_var, action_var) -
                                ys_computed_var))
    # q_loss_grad = theano.gradient.grad(q_loss, qf.get_params(trainable=True))

    f_train_qf = compile_function(
        inputs=input_vars + (ys_computed_var,),
        outputs=qf_loss,
        updates=adam(qf_loss, qf.get_params(trainable=True), learning_rate=1e-3)
    )

    policy_surr_loss = - TT.mean(
        qf.get_qval_sym(
            obs_var,
            policy.get_action_sym(obs_var),
            deterministic=True
        )
    )

    f_train_policy = compile_function(
        inputs=input_vars,
        outputs=policy_surr_loss,
        updates=adam(
            policy_surr_loss, policy.get_params(trainable=True), learning_rate=1e-4
        )
    )

    target_updates = []
    for param, target_param in zip(
        policy.get_params() + qf.get_params(),
        target_policy.get_params() + target_qf.get_params()
    ):
        target_updates.append(
            (target_param,
             param * options.soft_target_tau + target_param *
                (1 - options.soft_target_tau))
        )

    f_update_target = compile_function(
        inputs=tuple(),
        outputs=tuple(),
        updates=target_updates
    )

    itr = 0
    path_length = 0
    path_return = 0
    state, obs = mdp.reset()
    terminal = False

    for epoch in xrange(options.n_epochs):

        sample_path_returns = []
        for epoch_itr in pyprind.prog_bar(xrange(options.epoch_length)):

            if terminal or path_length > options.max_path_length:
                state, obs = mdp.reset()
                es.episode_reset()
                policy.episode_reset()
                sample_path_returns.append(path_return)
                path_length = 0
                path_return = 0

            action = es.get_action(itr, obs, policy)
            # print action
            next_state, next_obs, reward, terminal = mdp.step(state, action)
            mdp.plot()
            path_length += 1
            path_return += reward

            if path_length > options.max_path_length:
                terminal = True

            pool.add_sample(obs, action, reward, terminal)
            state, obs = next_state, next_obs

            if pool.size > options.min_pool_size:
                batch = pool.random_batch(options.batch_size)

                obses, actions, rewards, next_obses, terminals = extract(
                    batch,
                    "observations", "actions", "rewards", "next_observations",
                    "terminals"
                )

                inputs = (obses, actions, next_obses, rewards, terminals)

                ys = f_ys(*inputs)

                f_train_qf(*(inputs + (ys,)))
                f_train_policy(*inputs)
                f_update_target()

                itr += 1

        print "average sample path return:", np.mean(sample_path_returns)
        print "best sample path return:", np.max(sample_path_returns)
        print "worst sample path return:", np.min(sample_path_returns)
        print "std sample path return:", np.std(sample_path_returns)

        # evaluate policy

        if pool.size > options.min_pool_size:
            parallel_sampler.request_samples(
                policy_params=policy.get_param_values(),
                max_samples=options.eval_samples,
                max_path_length=options.max_path_length
            )

            paths = sum(parallel_sampler.run_map(worker_collect_paths), [])

            print "average reward:", np.mean([sum(p["rewards"]) for p in paths])
            print "average action:", np.mean(np.square(np.concatenate(
                [path["actions"] for path in paths]
            )))
            #    [sum(p["rewards"]) for p in paths])
