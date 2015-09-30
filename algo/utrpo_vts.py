from .utrpo import UTRPO
from mdp import ProxyMDP
from policy import DiscreteProxyPolicy
import theano.tensor as T
import itertools


class MDP_VTS(ProxyMDP):

    def __init__(self, base_mdp, time_scales):
        super(MDP_VTS, self).__init__(base_mdp)
        self._time_scales = time_scales

    def step(self, states, action_indices):
        next_states = []
        obs = []
        rewards = []
        dones = []
        steps = []
        n_base_actions = len(self._base_mdp.action_dims)
        for state, base_action_vts, scale_action in zip(
                states,
                zip(*action_indices[:-1]),
                action_indices[-1]):
            base_action = list(base_action_vts[n_base_actions*scale_action:n_base_actions*(scale_action+1)])
            # sometimes, the mdp will support the repeat mechanism which saves the time required to obtain intermediate observations (ram / images)
            if self._base_mdp.support_repeat:
                next_state, ob, reward, done, step = self._base_mdp.step_single(state, base_action, repeat=self._time_scales[scale_action])
            else:
                reward = 0
                step = 0
                next_state = state
                for _ in xrange(self._time_scales[scale_action]):
                    next_state, ob, step_reward, done, per_step = self._base_mdp.step_single(next_state, base_action)
                    reward += step_reward
                    step += per_step
                    if done:
                        break
            # experiment with counter the effect
            next_states.append(next_state)
            obs.append(ob)
            rewards.append(reward)
            dones.append(done)
            steps.append(step)
        return next_states, obs, rewards, dones, steps

        
# Unconstrained TRPO with variable time scales
class UTRPO_VTS(UTRPO):

    def __init__(self, time_scales=[4,16,64], *args, **kwargs):
        super(UTRPO_VTS, self).__init__(*args, **kwargs)
        self._time_scales = time_scales

    def transform_gen_mdp(self, gen_mdp):
        def gen_mdp_new():
            return MDP_VTS(gen_mdp(), self._time_scales)
        return gen_mdp_new

    def transform_gen_policy(self, gen_policy):
        def gen_policy_new(observation_shape, action_dims, input_var):
            scaled_action_dims = action_dims * 3 + [len(self._time_scales)]
            return gen_policy(
                    observation_shape, scaled_action_dims, input_var
                )
        return gen_policy_new

    def new_surrogate_obj(
            self, policy, input_var, Q_est_var, pi_old_vars, action_vars,
            lambda_var):
        Na = (len(policy.action_dims) - 1) / 3
        probs_vars = policy.probs_vars # ((3*Da+1) * N * Di
        # Model: p(a,t|s) = p(t|s)p(a|t,s)
        # KL(p_a,t(*|s)||q_a,t(*|s)) = KL(p_t(*|s)||q_t(*|s)) + E(KL(p_a(*|s,t)||q_a(*|s,t)))
        # Surrog_loss = p(t|s)p(a|t,s) / p_old(t|s)p_old(a|t,s) * Q(s,(a,t)) + KL...
        N_var = input_var.shape[0]
        mean_kl = None
        lr = None

        t_probs_var = probs_vars[-1]
        t_pi_old_var = pi_old_vars[-1]
        t_action_var = action_vars[-1]
        # This is the KL(p_t(*|s)||q_t(*|s)) term
        mean_kl = T.mean(
            T.sum(
                t_pi_old_var * (
                    T.log(t_pi_old_var + 1e-6) - T.log(t_probs_var + 1e-6)
                ),
                axis=1
            ))

        for tid in range(len(self._time_scales)):
            for probs_var, pi_old_var in zip(
                    probs_vars[tid*Na:(tid+1)*Na],
                    pi_old_vars[tid*Na:(tid+1)*Na]):

                mean_kl = mean_kl + T.mean(
                    t_pi_old_var[:,tid] * T.sum(
                        pi_old_var * (
                            T.log(pi_old_var + 1e-6) - T.log(probs_var + 1e-6)
                        ),
                        axis=1
                    ))

        # This is the p(t|s) / p_old(t|s) term
        lr_t = t_probs_var[T.arange(N_var), t_action_var] / (t_pi_old_var[T.arange(N_var), t_action_var] + 1e-6)
        # The variables are laid out like a1|t=1, a2|t=1, ..., a1|t=2, a2|t=2, ..., an|t=1, ..., an|t=k
        lr_a = 0
        for idx, probs_var, pi_old_var, action_var in zip(
                itertools.count(), probs_vars[:-1], pi_old_vars[:-1], action_vars[:-1]):
            tid = idx / Na
            aid = idx % Na
            pi_old_selected = pi_old_var[T.arange(N_var), action_var]
            pi_selected = probs_var[T.arange(N_var), action_var]
            teq = T.eq(t_action_var, tid)
            lr_a += teq * pi_selected / (pi_old_selected + 1e-6)
        # formulate as a minimization problem
        surrogate_loss = - T.mean(lr_t * lr_a * Q_est_var)
        surrogate_obj = surrogate_loss + lambda_var * mean_kl
        return surrogate_obj, surrogate_loss, mean_kl
