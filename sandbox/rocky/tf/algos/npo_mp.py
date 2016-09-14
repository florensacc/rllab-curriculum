

from rllab.misc import ext
from rllab.misc.overrides import overrides
import rllab.misc.logger as logger
from rllab.sampler.parallel_sampler import _get_scoped_G
from sandbox.rocky.tf.algos.batch_polopt import BatchPolopt
from sandbox.rocky.tf.misc import tensor_utils
import tensorflow as tf
import pickle
from rllab.sampler.stateful_pool import singleton_pool
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer_mp import ConjugateGradientOptimizerMP


class NPOMP(BatchPolopt):
    """
    Multiprocessing version of Natural Policy Optimization.
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            step_size=0.01,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = ConjugateGradientOptimizerMP(**optimizer_args)
        assert isinstance(optimizer, ConjugateGradientOptimizerMP)
        self.optimizer = optimizer
        self.step_size = step_size
        super(NPOMP, self).__init__(**kwargs)


    @overrides
    def init_opt(self):
        """
        BatchPolopt calls init_opt in __init__, but we need to get around that since the stubbed CGMP cannot use its __args, __kwargs to create copies
        """
        self.init_opt_done = False
        return dict()

    def init_opt_later(self):
        """
        Called in self.optimize_policy() to build the tensorflow graphs
        """
        copy_optimizer(self.optimizer.worker_copy())
        singleton_pool.run_each(
            _worker_init_opt,
            [(self.step_size, None,)] * singleton_pool.n_parallel
        )
        self.optimizer._target = self.policy
        self.optimizer._max_constraint_val = self.step_size
        self.optimizer._constraint_name = "mean_kl"
        self.init_opt_done = True

    @overrides
    def optimize_policy(self, itr, samples_data):
        # build tensorflow graphs
        if not self.init_opt_done:
            self.init_opt_later()

        # define optimizer inputs
        all_input_values = tuple(ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        ))
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        all_input_values += tuple(state_info_list) + tuple(dist_info_list)
        if self.policy.recurrent:
            all_input_values += (samples_data["valids"],)

        # tell the optimizer to load inputs to different workers
        self.optimizer.prepare_optimization(all_input_values)

        logger.log("Computing loss and KL before")
        loss_before, mean_kl_before = self.optimizer.loss_constraint_val()

        logger.log("Optimizing")
        self.optimizer.optimize(loss_before=loss_before)

        logger.log("Computing loss and KL after")
        loss_after, mean_kl_after = self.optimizer.loss_constraint_val()

        logger.record_tabular('LossBefore', loss_before)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('MeanKLBefore', mean_kl_before)
        logger.record_tabular('MeanKL', mean_kl_after)
        logger.record_tabular('dLoss', loss_before - loss_after)
        return dict()

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )

def _worker_copy_optimizer(G, optimizer_pkl, scope=None):
    G = _get_scoped_G(G, scope)
    G.optimizer = pickle.loads(optimizer_pkl)

# May need to modify paralle_sampler to delete G.optimizer for each worker
# def _worker_terminate_task

def copy_optimizer(optimizer, scope=None):
    logger.log("Copying optimizers...")
    if singleton_pool.n_parallel > 1:
        singleton_pool.run_each(
            _worker_copy_optimizer,
            [(pickle.dumps(optimizer),scope)] * singleton_pool.n_parallel
        )
    else:
        G = _get_scoped_G(singleton_pool.G, scope)
        G.optimizer = optimizer
    logger.log("Copied")

def _worker_init_opt(G, step_size, scope):
    G = _get_scoped_G(G, scope)
    # asssume that policies are assigned to workers
    is_recurrent = int(G.policy.recurrent)
    obs_var = G.env.observation_space.new_tensor_variable(
        'obs',
        extra_dims=1 + is_recurrent,
    )
    action_var = G.env.action_space.new_tensor_variable(
        'action',
        extra_dims=1 + is_recurrent,
    )
    advantage_var = tensor_utils.new_tensor(
        'advantage',
        ndim=1 + is_recurrent,
        dtype=tf.float32,
    )
    dist = G.policy.distribution

    old_dist_info_vars = {
        k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name='old_%s' % k)
        for k, shape in dist.dist_info_specs
        }
    old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

    state_info_vars = {
        k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name=k)
        for k, shape in G.policy.state_info_specs
        }
    state_info_vars_list = [state_info_vars[k] for k in G.policy.state_info_keys]

    if is_recurrent:
        valid_var = tf.placeholder(tf.float32, shape=[None, None], name="valid")
    else:
        valid_var = None

    dist_info_vars = G.policy.dist_info_sym(obs_var, state_info_vars)
    kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
    lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)
    if is_recurrent:
        mean_kl = tf.reduce_sum(kl * valid_var) / tf.reduce_sum(valid_var)
        surr_loss = - tf.reduce_sum(lr * advantage_var * valid_var) / tf.reduce_sum(valid_var)
    else:
        mean_kl = tf.reduce_mean(kl)
        surr_loss = - tf.reduce_mean(lr * advantage_var)

    input_list = [
                     obs_var,
                     action_var,
                     advantage_var,
                 ] + state_info_vars_list + old_dist_info_vars_list
    if is_recurrent:
        input_list.append(valid_var)

    G.optimizer.update_opt(
        loss=surr_loss,
        target=G.policy,
        leq_constraint=(mean_kl, step_size),
        inputs=input_list,
        constraint_name="mean_kl"
    )
