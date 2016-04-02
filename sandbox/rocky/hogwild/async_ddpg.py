from rllab.algos.base import RLAlgorithm
from rllab.core.serializable import Serializable
from sandbox.rocky.hogwild.shared_policy import SharedPolicy
from sandbox.rocky.hogwild.shared_q_function import SharedQFunction
from functools import partial
from rllab.misc import ext
import multiprocessing as mp
import lasagne
import theano.tensor as TT


def start_worker(algo, worker_id, pipe):
    algo.worker_train(worker_id=worker_id, pipe=pipe)


def parse_update_method(update_method, **kwargs):
    if update_method == 'adam':
        return partial(lasagne.updates.adam, **ext.compact(kwargs))
    elif update_method == 'sgd':
        return partial(lasagne.updates.sgd, **ext.compact(kwargs))
    else:
        raise NotImplementedError


class AsyncDDPG(RLAlgorithm, Serializable):
    """
    Asynchronous Deep Deterministic Policy Gradient.
    """

    def __init__(
            self,
            env,
            policy,
            qf,
            es,
            target_policy=None,
            target_qf=None,
            n_workers=1,
            qf_weight_decay=0.,
            qf_update_method='adam',
            qf_learning_rate=1e-3,
            policy_weight_decay=0,
            policy_update_method='adam',
            policy_learning_rate=1e-4,
    ):
        # Make sure all variables are properly initialized when serialized to worker copies
        if not isinstance(policy, SharedPolicy):
            policy = SharedPolicy(policy)
        if not isinstance(qf, SharedQFunction):
            qf = SharedQFunction(qf)
        if target_policy is None:
            target_policy = policy.new_mem_copy()
        if target_qf is None:
            target_qf = qf.new_mem_copy()
        Serializable.quick_init(self, locals())
        self.env = env
        self.policy = policy
        self.qf = qf
        self.target_policy = target_policy
        self.target_qf = target_qf
        self.es = es
        self.n_workers = n_workers

        self.qf_weight_decay = qf_weight_decay
        self.qf_update_method = \
            parse_update_method(
                qf_update_method,
                learning_rate=qf_learning_rate,
            )
        self.qf_learning_rate = qf_learning_rate
        self.policy_weight_decay = policy_weight_decay
        self.policy_update_method = \
            parse_update_method(
                policy_update_method,
                learning_rate=policy_learning_rate,
            )
        self.policy_learning_rate = policy_learning_rate
        self.opt_info = None

    def train(self):
        """
        Start the training procedure on the master process. It launches several worker processes.
        """
        processes = []
        for id in xrange(self.n_workers):
            pipe = mp.Pipe()
            p = mp.Process(target=start_worker, args=(self, id, pipe))
            p.start()
            processes.append(p)

    def worker_init_opt(self):
        """
        Initialize the computation graph on the worker process
        """
        # y need to be computed first
        obs = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1,
        )

        # The yi values are computed separately as above and then passed to
        # the training functions below
        action = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1,
        )
        yvar = TT.vector('ys')

        qf_weight_decay_term = 0.5 * self.qf_weight_decay * \
                               sum([TT.sum(TT.square(param)) for param in
                                    self.qf.get_params(regularizable=True)])

        qval = self.qf.get_qval_sym(obs, action)

        qf_loss = TT.mean(TT.square(yvar - qval))
        qf_reg_loss = qf_loss + qf_weight_decay_term

        policy_weight_decay_term = 0.5 * self.policy_weight_decay * \
                                   sum([TT.sum(TT.square(param))
                                        for param in self.policy.get_params(regularizable=True)])
        policy_qval = self.qf.get_qval_sym(
            obs, self.policy.get_action_sym(obs),
            deterministic=True
        )
        policy_surr = -TT.mean(policy_qval)

        policy_reg_surr = policy_surr + policy_weight_decay_term

        qf_grads = TT.grad(qf_reg_loss, self.qf.get_params(trainable=True))
        policy_grads = TT.grad(policy_reg_surr, self.policy.get_params(trainable=True))

        f_qf_grads = ext.compile_function(
            inputs=[yvar, obs, action],
            outputs=qf_grads,
        )

        f_policy_grads = ext.compile_function(
            inputs=[obs],
            outputs=policy_grads,
        )

        self.opt_info = dict(
            f_qf_grads=f_qf_grads,
            f_policy_grads=f_policy_grads,
        )

    def worker_train(self, worker_id, pipe):
        """
        Training procedure on worker processes
        :param worker_id: ID assigned to the worker for logging purposes
        :param pipe: a multiprocessing.Pipe object for communicating with the master process
        """

        # Each worker needs to compile their own functions
        self.worker_init_opt()
        while True:

            pass
