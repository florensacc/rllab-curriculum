


# from rllab.sampler.stateful_pool import singleton_pool
from rllab.sampler import parallel_sampler
import multiprocessing as mp

parallel_sampler.initialize(n_parallel=mp.cpu_count())

from sandbox.rocky.hrl.policies.stochastic_gru_policy import StochasticGRUPolicy
from sandbox.rocky.hrl.bonus_evaluators.discrete_bonus_evaluator import DiscreteBonusEvaluator, MODES
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.hrl.envs.perm_grid_env import PermGridEnv
from rllab.misc.special import to_onehot
from rllab.misc import logger
from rllab.regressors.gaussian_mlp_regressor import GaussianMLPRegressor
from rllab.optimizers.first_order_optimizer import FirstOrderOptimizer
from rllab.core.parameterized import Parameterized
import numpy as np
import joblib
import lasagne.layers as L
import theano.tensor as TT


class JointParameterized(Parameterized):
    def __init__(self, components):
        super(JointParameterized, self).__init__()
        self.components = components

    def get_params_internal(self, **tags):
        return [param for comp in self.components for param in comp.get_params_internal(**tags)]


bottleneck_dim = 7

grid_size = 10  # 10

# def config():
env = PermGridEnv(
    size=grid_size,
    n_objects=4,
    object_seed=0,
    random_restart=True
)

policy = StochasticGRUPolicy(
    env_spec=env.spec,
    n_subgoals=4,
    action_bottleneck_dim=bottleneck_dim,
    hidden_bottleneck_dim=bottleneck_dim,
    bottleneck_dim=bottleneck_dim,
    use_bottleneck=True,
    deterministic_bottleneck=True,
    bottleneck_nonlinear=True,
    separate_bottlenecks=True,
    use_decision_nodes=True,#False,
    bottleneck_hidden_sizes=tuple(),
)
n_training_paths = 10000  # 0#0

parallel_sampler.populate_task(env=env, policy=policy)


# return locals()


# Let's try to use supervised learning to train a policy

# first, let's generate some training data..

def gen_sup_trajectory():
    """
    :type env: PermGridEnv
    """
    obs = env.reset()
    totrew = 0
    observations = []
    actions = []
    hidden_states = []
    prev_hidden_states = []
    action_bottlenecks = []
    agent_pos_space = env.observation_space.components[0]
    hidden_state = to_onehot(0, env.n_objects)
    while True:
        agent_pos, visit_order, visited_flags = obs
        # determine the object to visit
        obj_to_visit = visit_order[np.sum(visited_flags)]
        visit_obj_pos = env.object_positions[obj_to_visit]
        # determine the action to take
        if visit_obj_pos[0] < agent_pos[0]:
            action = env.action_from_direction("up")
        elif visit_obj_pos[0] > agent_pos[0]:
            action = env.action_from_direction("down")
        elif visit_obj_pos[1] < agent_pos[1]:
            action = env.action_from_direction("left")
        elif visit_obj_pos[1] > agent_pos[1]:
            action = env.action_from_direction("right")
        else:
            raise NotImplementedError
        observations.append(env.observation_space.flatten(obs))
        actions.append(env.action_space.flatten(action))
        prev_hidden_states.append(hidden_state)
        hidden_state = to_onehot(obj_to_visit, env.n_objects)
        hidden_states.append(hidden_state)
        action_bottlenecks.append(agent_pos_space.flatten(agent_pos))
        obs, rew, done, _ = env.step(action)
        totrew += rew
        if done:
            break
    return dict(
        observations=np.asarray(observations),
        actions=np.asarray(actions),
        hidden_states=np.asarray(hidden_states),
        prev_hidden_states=np.asarray(prev_hidden_states),
        action_bottlenecks=np.asarray(action_bottlenecks)
    )


if __name__ == "__main__":
    logger.log("Generating training data...")
    with joblib.Parallel(n_jobs=-1) as parallel:
        paths = parallel(joblib.delayed(gen_sup_trajectory)() for _ in range(n_training_paths))
    logger.log("Generated")

    all_obs = np.concatenate([p["observations"] for p in paths])
    all_actions = np.concatenate([p["actions"] for p in paths])
    all_hidden_states = np.concatenate([p["hidden_states"] for p in paths])
    all_prev_hidden_states = np.concatenate([p["prev_hidden_states"] for p in paths])
    all_action_bottlenecks = np.concatenate([p["action_bottlenecks"] for p in paths])

    train_action_var = TT.matrix("train_actions")
    train_hiddens_var = TT.matrix("train_hiddens")
    train_action_bottlenecks_var = TT.matrix("action_bottlenecks")

    # apply random projection to the bottlenecks
    bottleneck_embedding = np.random.standard_normal(size=(all_action_bottlenecks.shape[1], bottleneck_dim))

    all_action_bottleneck_embeddings = np.tanh(all_action_bottlenecks.dot(bottleneck_embedding))

    # Form the objectives for different components

    action_bottleneck_var = L.get_output(policy.l_action_bottleneck_out)
    hidden_bottleneck_var = L.get_output(policy.l_hidden_bottleneck_out)

    action_prob_var = policy.action_prob_sym(
        action_obs_var=action_bottleneck_var,
        hidden_var=policy.l_hidden.input_var
    )

    hidden_prob_var = policy.hidden_prob_sym(
        hidden_obs_var=hidden_bottleneck_var,
        prev_hidden_var=policy.l_prev_hidden.input_var,
    )

    # cross entropy loss for the actions and hidden states
    #
    # TODO
    # in addition, we'd want the bottleneck states to match. Since we use a continuous bottleneck representation,
    # we constrain it so that it should be predictive from only the ground truth action bottleneck state

    # action_bottleneck_regressor = GaussianMLPRegressor(
    #     input_shape=(all_action_bottlenecks.shape[1],),
    #     output_dim=bottleneck_dim,
    #     learn_std=False,
    #     hidden_nonlinearity=TT.tanh,
    # )

    loss = - TT.mean(
        TT.sum(train_action_var * TT.log(action_prob_var + 1e-8), axis=1) +
        TT.sum(train_hiddens_var * TT.log(hidden_prob_var + 1e-8), axis=1) +
        (-TT.mean(TT.square(train_action_bottlenecks_var - action_bottleneck_var)))
    )
    # loss = - TT.mean(
    #     (-TT.mean(TT.square(train_action_bottlenecks_var - action_bottleneck_var)))
    # )


    def log_progress(loss, **kwargs):
        logger.log("Loss: %f" % loss)


    optimizer = FirstOrderOptimizer(
        verbose=True,
        max_epochs=10#0
    )
    input_vars = [
        policy.l_raw_obs.input_var,
        policy.l_prev_hidden.input_var,
        policy.l_hidden.input_var,
        train_action_var,
        train_hiddens_var,
        train_action_bottlenecks_var
    ]
    inputs = [
        all_obs,
        all_prev_hidden_states,
        all_hidden_states,
        all_actions,
        all_hidden_states,
        all_action_bottleneck_embeddings
    ]
    optimizer.update_opt(loss, target=policy, inputs=input_vars)
    optimizer.optimize(inputs, callback=log_progress)

    # optimizer.update_opt(loss2, target=policy, inputs=input_vars)
    # optimizer.optimize(inputs, callback=log_progress)

    test_paths = parallel_sampler.sample_paths(
        policy_params=policy.get_param_values(),
        max_samples=20000,
        max_path_length=100,
    )

    bonus_evaluator = DiscreteBonusEvaluator(
        env_spec=env.spec,
        policy=policy,
        mode=MODES.MODE_BOTTLENECK_ONLY,
        regressor_args=dict(
            use_trust_region=False,
            step_size=0.01,
            hidden_nonlinearity=TT.tanh,
            hidden_sizes=(100, 100),
        ),
        use_exact_regressor=True,
        exact_stop_gradient=True,
        exact_entropy=False,
    )
    bonus_evaluator.fit(test_paths)
    bonus_evaluator.log_diagnostics(test_paths)

    avg_return = np.mean([sum(path["rewards"]) for path in test_paths])

    logger.record_tabular("AverageReturn", avg_return)
    # logger.record_tabular("AverageDiscountedReturn", avg_return)

    logger.dump_tabular()

    import ipdb;

    ipdb.set_trace()
