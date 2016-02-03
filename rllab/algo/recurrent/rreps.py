import theano.tensor as TT
from rllab.misc import logger, autoargs
from rllab.misc.overrides import overrides
from rllab.misc.ext import extract, compile_function, new_tensor
from rllab.algo.recurrent.recurrent_batch_polopt import RecurrentBatchPolopt
import numpy as np
from rllab.misc.tensor_utils import flatten_tensors, pad_tensor
from pydoc import locate
  
  
class RREPS(RecurrentBatchPolopt):
    """
    Recurrent Relative Entropy Policy Search (REPS).
    """
  
    @autoargs.inherit(RecurrentBatchPolopt.__init__)
    @autoargs.arg("epsilon", type=float,
                  help="Max KL divergence between new policy and old policy.")
    @autoargs.arg("L2_reg_dual", type=float,
                  help="Dual regularization")
    @autoargs.arg("L2_reg_loss", type=float,
                  help="Loss regularization")
    @autoargs.arg("max_opt_itr", type=int,
                  help="Maximum number of batch optimization iterations.")
    @autoargs.arg("optimizer", type=str,
                  help="Module path to the optimizer. It must support the "
                  "same interface as scipy.optimize.fmin_l_bfgs_b")
    def __init__(
            self,
            epsilon=0.1,
            L2_reg_dual=1e-7,
            L2_reg_loss=0.,
            max_opt_itr=50,
            optimizer='scipy.optimize.fmin_l_bfgs_b',
            **kwargs):
        super(RREPS, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.L2_reg_dual = L2_reg_dual
        self.L2_reg_loss = L2_reg_loss
        self.max_opt_itr = max_opt_itr
        self.optimizer = locate(optimizer)
  
    @overrides
    def init_opt(self, mdp, policy, baseline):
  
        # Init dual param values
        self.param_eta = 15.
        # Adjust for linear feature vector.
        self.param_v = np.random.randn(mdp.observation_shape[0] * 2 + 4)
  
        # Theano vars
        observations = new_tensor(
            'observations',
            ndim=2 + len(mdp.observation_shape),
            dtype=mdp.observation_dtype
        )
        action_var = new_tensor(
            'action',
            ndim=3,
            dtype=mdp.action_dtype
        )
        rewards = TT.matrix('rewards',
                            dtype=TT.config.floatX)  # @UndefinedVariable
        feat_diff = new_tensor(
            'feat_diff',
            ndim=3,
            dtype=TT.config.floatX
        )  # @UndefinedVariable

        valid_var = TT.matrix('valid')

        param_v = TT.vector('param_v',
                            dtype=TT.config.floatX)  # @UndefinedVariable
        param_eta = TT.scalar('eta',
                              dtype=TT.config.floatX)  # @UndefinedVariable
  
        # Policy stuff
        # log of the policy dist
        log_prob = policy.get_log_prob_sym(observations, action_var)
  
        # Symbolic sample Bellman error
        delta_v = rewards + TT.dot(feat_diff, param_v)
  
        # Policy loss (negative because we minimize)
        loss = - TT.sum(log_prob * TT.exp(
            delta_v / param_eta - TT.max(delta_v / param_eta)
        ) * valid_var) / TT.sum(valid_var)
        # Add regularization to loss.
        reg_params = policy.get_params(regularizable=True)
        loss += self.L2_reg_loss * TT.sum([
            TT.mean(TT.square(param)) for param in reg_params]) / \
            len(reg_params)
  
        # Policy loss gradient.
        loss_grad = TT.grad(
            loss, policy.get_params(trainable=True))
  
        input = [rewards, observations, feat_diff,
                 action_var, valid_var, param_eta, param_v]
        f_loss = compile_function(
            inputs=input,
            outputs=loss,
        )
        f_loss_grad = compile_function(
            inputs=input,
            outputs=loss_grad,
        )
  
        # Debug prints
        old_pdist = new_tensor(
            'old_pdist',
            ndim=3,
            dtype=TT.config.floatX
        )
        pdist = policy.get_pdist_sym(observations, action_var)
        mean_kl = TT.sum(policy.kl(old_pdist, pdist) * valid_var) / \
            TT.sum(valid_var)
        f_kl = compile_function(
            inputs=[observations, action_var, old_pdist, valid_var],
            outputs=mean_kl,
        )
  
        # Dual stuff
        # Symbolic dual
        dual = param_eta * self.epsilon + param_eta * \
            TT.log(
                TT.sum(
                    TT.exp(
                        delta_v / param_eta - TT.max(delta_v / param_eta)
                    ) * valid_var
                ) / TT.sum(valid_var)
            ) + param_eta * TT.max(delta_v / param_eta)
        # Add L2 regularization.
        dual += self.L2_reg_dual * \
            (TT.mean(param_eta**2) +
             TT.mean((1 / param_eta)**2) + TT.mean(param_v**2))
  
        # Symbolic dual gradient
        dual_grad = TT.grad(cost=dual, wrt=[param_eta, param_v])
  
        # Eval functions.
        f_dual = compile_function(
            inputs=[rewards, feat_diff, valid_var, param_eta, param_v],
            outputs=dual
        )
        f_dual_grad = compile_function(
            inputs=[rewards, feat_diff, valid_var, param_eta, param_v],
            outputs=dual_grad
        )
  
        return dict(
            f_loss_grad=f_loss_grad,
            f_loss=f_loss,
            f_dual=f_dual,
            f_dual_grad=f_dual_grad,
            f_kl=f_kl
        )
  
    def _features(self, path):
        o = np.clip(path["observations"], -10, 10)
        l = len(path["rewards"])
        al = np.arange(l).reshape(-1, 1) / 100.0
        return np.concatenate([o, o**2, al, al**2, al**3, np.ones((l, 1))], axis=1)
  
    @overrides
    def optimize_policy(self, itr, policy, samples_data, opt_info):
  
        # Init vars
        actions = samples_data['actions']
        observations = samples_data['observations']
        # Compute sample Bellman error.
        feat_diff = []
        for path in samples_data['paths']:
            feats = self._features(path)
            feats = np.vstack([feats, np.zeros(feats.shape[1])])
            feat_diff.append(feats[1:] - feats[:-1])

        # Get all paths.
        paths = samples_data["paths"]
        # Max path length to construct mask to handle input sequences of
        # varying length.
        max_path_length = max([len(path["advantages"]) for path in paths])

        # Rewards should be set to for padding.
        rewards = [path['rewards'] for path in paths]
        rewards = [pad_tensor(r, max_path_length, 0) for r in rewards]

        feat_diff = [pad_tensor(fd, max_path_length, fd[0])
                     for fd in feat_diff]
        valids = samples_data["valids"]
  
        #################
        # Optimize dual #
        #################
  
        # Here we need to optimize dual through BFGS in order to obtain \eta
        # value. Initialize dual function g(\theta, v). \eta > 0
        # First eval delta_v
        f_dual = opt_info['f_dual']
        f_dual_grad = opt_info['f_dual_grad']
  
        # Set BFGS eval function
        def eval_dual(input):
            param_eta = input[0]
            param_v = input[1:]
            val = f_dual(rewards, feat_diff, valids, param_eta, param_v)
            return val.astype(np.float64)
  
        # Set BFGS gradient eval function
        def eval_dual_grad(input):
            param_eta = input[0]
            param_v = input[1:]
            grad = f_dual_grad(rewards, feat_diff, valids, param_eta, param_v)
            eta_grad = np.float(grad[0])
            v_grad = grad[1]
            return np.hstack([eta_grad, v_grad])
  
        # Initial BFGS parameter values.
        x0 = np.hstack([self.param_eta, self.param_v])
  
        # Set parameter boundaries: \eta>0, v unrestricted.
        bounds = [(-np.inf, np.inf) for _ in x0]
        bounds[0] = (0., np.inf)
  
        # Optimize through BFGS
        logger.log('optimizing dual')
        eta_before = x0[0]
        dual_before = eval_dual(x0)
        params_ast, _, _ = self.optimizer(
            func=eval_dual, x0=x0,
            fprime=eval_dual_grad,
            bounds=bounds,
            maxiter=self.max_opt_itr,
            disp=0
        )
        dual_after = eval_dual(params_ast)
  
        # Optimal values have been obtained
        self.param_eta = params_ast[0]
        self.param_v = params_ast[1:]
  
        ###################
        # Optimize policy #
        ###################
        cur_params = policy.get_param_values(trainable=True)
        f_loss = opt_info["f_loss"]
        f_loss_grad = opt_info['f_loss_grad']
        input = [rewards, observations, feat_diff,
                 actions, valids, self.param_eta, self.param_v]
  
        # Set loss eval function
        def eval_loss(params):
            policy.set_param_values(params, trainable=True)
            val = f_loss(*input)
            return val.astype(np.float64)
  
        # Set loss gradient eval function
        def eval_loss_grad(params):
            policy.set_param_values(params, trainable=True)
            grad = f_loss_grad(*input)
            flattened_grad = flatten_tensors(map(np.asarray, grad))
            return flattened_grad.astype(np.float64)
  
        loss_before = eval_loss(cur_params)
        logger.log('optimizing policy')
        params_ast, _, _ = self.optimizer(
            func=eval_loss, x0=cur_params,
            fprime=eval_loss_grad,
            disp=0,
            maxiter=self.max_opt_itr
        )
        opt_params = policy.get_param_values(trainable=True)
        loss_after = eval_loss(opt_params)
  
        f_kl = opt_info['f_kl']
        old_pdist = samples_data['pdists']
        mean_kl = f_kl(observations, actions, old_pdist, valids).astype(np.float64)
  
        logger.log('eta %f -> %f' % (eta_before, self.param_eta))
  
        logger.record_tabular("EtaBefore", eta_before)
        logger.record_tabular("EtaAfter", self.param_eta)
        logger.record_tabular("LossBefore", loss_before)
        logger.record_tabular("LossAfter", loss_after)
        logger.record_tabular('DualBefore', dual_before)
        logger.record_tabular('DualAfter', dual_after)
        logger.record_tabular('MeanKL', mean_kl)
  
        return opt_info
  
    @overrides
    def get_itr_snapshot(self, itr, mdp, policy, baseline, samples_data,
                         opt_info):
        return dict(
            itr=itr,
            policy=policy,
            baseline=baseline,
            mdp=mdp,
        )
