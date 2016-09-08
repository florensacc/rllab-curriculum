import rllab.misc.logger as logger
from rllab.core.serializable import Serializable
from rllab.misc import special, krylov
import tensorflow as tf
from collections import defaultdict
import numpy as np
from sandbox.john import tf_util as U, common

LOG_SETTINGS = dict(with_prefix=False, with_timestamp=False)
def _log(s): logger.log(s, **LOG_SETTINGS)

class TRPO(Serializable):
    def __init__(
        self,
        venv,
        policy,
        baseline,
        horizon=20,
        discount=0.99,
        gae_lambda=1.0,
        max_kl=0.01,
        cg_damping=1e-2,
    ):
        Serializable.quick_init(self, locals())

        self.venv = venv
        self.policy = policy
        self.baseline = baseline
        self.horizon = horizon
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.max_kl = max_kl
        self.cg_damping = cg_damping

        ob_space = venv.observation_space
        batch_size = venv.num_envs

        # Build graph
        # ----------------------------------------        
        self.syms = defaultdict(list)
        for t in range(horizon):
            ob = common.space2variable(ob_space, name = "ob%i"%t, prepend_shape=[batch_size])
            ac, psi = policy.act(ob)
            self.syms["psi"].append(psi)
            adv = tf.placeholder(tf.float32, shape=[batch_size], name="adv%i"%t)
            oldpsi = tf.placeholder(tf.float32, shape=psi.get_shape(), name="oldpsi%i"%t)
            self.syms["oldpsi"].append(oldpsi)
            self.syms["ob"].append(ob)
            self.syms["ac"].append(ac)
            self.syms["ac2"].append(tf.placeholder(ac.dtype, ac.get_shape()))
            self.syms["adv"].append(adv)
            self.syms["new"].append(tf.placeholder(tf.int32, shape=[batch_size], name="new%i"%t))

        policy_vars = policy.vars
        self.set_policy_from_flat = U.SetFromFlat(policy_vars)
        self.get_policy_flat = U.GetFlat(policy_vars)


        catoldpsi = tf.concat(0, self.syms["oldpsi"])
        catpsi = tf.concat(0, self.syms["psi"])
        avg_kl = U.mean(policy.compute_kl(catoldpsi, catpsi))
        # avg_kl_local = U.mean(policy.compute_kl(tf.stop_gradient(catpsi), catpsi))
        avg_kl_local = avg_kl
        logpofa = policy.compute_loglik(catpsi, tf.concat(0, self.syms["ac2"]))
        logpoldofa = policy.compute_loglik(catoldpsi, tf.concat(0, self.syms["ac2"]))
        avg_ent = U.mean(policy.compute_entropy(catpsi))
        avg_surr = - U.mean((logpofa - logpoldofa) * tf.concat(0, self.syms["adv"]))

        num_params = np.sum([np.prod(U.var_shape(v)) for v in policy_vars])
        self.pg = U.flatgrad(avg_surr, policy_vars)        
        klgrads = tf.gradients(avg_kl_local, policy_vars)
        self.flat_tangent = tf.placeholder(tf.float32, shape=num_params)
        shapes = [[x.value for x in var.get_shape()] for var in policy_vars]
        start = 0
        tangents = []
        for shape in shapes:
            size = np.prod(shape)
            tangents.append(U.reshape(self.flat_tangent[start:start+size], shape))
            start += size
        gvp = tf.add_n([U.sum(g*tangent) for (g, tangent) in zip(klgrads, tangents)]) #pylint: disable=E1111
        # Fisher-vector product
        self.fvp = U.flatgrad(gvp, policy_vars)

        self.losses = [avg_surr, avg_kl, avg_ent]
        self.loss_names = ["surr", "kl", "ent"]

        U.initialize()
        self.filt = common.ZFilter()

    def train(self, n_updates):
        batch_size = self.venv.num_envs
        ob_raw = self.venv.reset()
        ob = self.filt(ob_raw)
        new = np.ones(batch_size, bool)
        total_timesteps = 0
        total_episodes = 0
        rets = np.zeros(batch_size, 'float32')
        for i_update in range(n_updates):
            episode_returns = []
            setup = U.SESSION.partial_run_setup(
                self.syms["ac"] + self.syms["psi"],
                self.syms["ob"], 
                )
            rews = np.zeros((self.horizon, batch_size), 'float32')
            news = np.zeros((self.horizon, batch_size) , 'float32')
            obs = np.zeros((self.horizon,) + ob.shape, ob.dtype)
            feed = {}
            allacs = []


            for t in range(self.horizon):
                news[t] = new
                obs[t] = ob
                ob_sym = self.syms["ob"][t]
                ac, psi = U.SESSION.partial_run(setup, 
                    [self.syms["ac"][t], self.syms["psi"][t]], feed_dict={ob_sym : ob})
                feed[ob_sym] = ob
                feed[self.syms["ac2"][t]] = ac
                feed[self.syms["oldpsi"][t]] = psi
                feed[self.syms["new"][t]] = new
                ob_raw, rews[t], new = self.venv.step(ac)
                ob = self.filt(ob_raw)
                rets += rews[t]
                if total_timesteps > 0:
                    for i_env in np.flatnonzero(new):
                        episode_returns.append(rets[i_env])
                        rets[i_env] = 0
                total_timesteps += 1
                allacs.append(ac)

            # _log("Fitting baseline")
            rews1 = rews.copy()
            rews1[-1] += self.baseline.predict(ob)
            vtargs = common.discounted_future_sum(rews1, news, self.discount)
            vtargs = np.clip(vtargs, -10, 10) # XXX
            self.baseline.fit(obs.reshape(self.horizon*batch_size, -1), vtargs.reshape(-1))

            vpreds = np.zeros((self.horizon+1, batch_size), 'float32')
            vpreds[:self.horizon] = self.baseline.predict(obs.reshape(self.horizon*batch_size, -1)).reshape(self.horizon, batch_size)
            vpreds[self.horizon] = self.baseline.predict(ob)

            deltas = rews + self.discount * vpreds[1:] - vpreds[:-1]
            advs = common.discounted_future_sum(deltas, news, self.gae_lambda * self.discount)
            # Note that we demean advantages by computing mean at each timestep
            standardized_advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            for t in range(self.horizon):
                feed[self.syms["adv"][t]] = standardized_advs[t]

            # _log("Updating policy")    
            thprev = self.get_policy_flat()
            def fisher_vector_product(p):
                feed[self.flat_tangent] = p
                out = U.SESSION.run(self.fvp, feed) + self.cg_damping * p
                del feed[self.flat_tangent]
                return out
            losses_before = U.SESSION.run(self.losses, feed)
            g = U.SESSION.run(self.pg, feed)
            if np.allclose(g, 0):
                _log("got zero gradient. not updating")                
                # import IPython; IPython.embed();
            else:
                stepdir = krylov.cg(fisher_vector_product, -g)
                shs = .5*stepdir.dot(fisher_vector_product(stepdir))
                lm = np.sqrt(shs / self.max_kl)
                logger.log("lagrange multiplier: %g. gnorm: %g"%(lm, np.linalg.norm(g)))
                fullstep = stepdir / lm
                neggdotstepdir = -g.dot(stepdir) #pylint: disable=E1101
                def loss(th):
                    self.set_policy_from_flat(th)
                    loss, kl =  U.SESSION.run(self.losses[0:2], feed)
                    return loss + 1e10 * (kl > self.max_kl * 1.5)
                success, theta = _linesearch(loss, thprev, fullstep, neggdotstepdir/lm)
                _log("success: %g"%success)
                self.set_policy_from_flat(theta)
            losses_after = U.SESSION.run(self.losses, feed)

            for (name, beforeval, afterval) in zip(self.loss_names, losses_before, losses_after):
                logger.record_tabular("%s_Before"%name, beforeval)
                logger.record_tabular("%s_After"%name, afterval)

            vtargs = vpreds[:-1] + advs
            losses_dict = dict(list(zip(self.loss_names, losses_after)))
            ent = losses_dict["ent"]
            total_episodes += len(episode_returns)         
            logger.record_tabular("Iteration", i_update)
            logger.record_tabular("NumTrajs", len(episode_returns))
            ok = len(episode_returns) > 0
            logger.record_tabular("MinReturn", np.min(episode_returns) if ok else np.nan)
            logger.record_tabular("MaxReturn", np.max(episode_returns) if ok else np.nan)
            logger.record_tabular("MeanReturn", np.mean(episode_returns) if ok else np.nan)
            logger.record_tabular("StdReturn", np.std(episode_returns) if ok else np.nan)
            logger.record_tabular("Entropy", ent)
            logger.record_tabular("Perplexity", np.exp(ent))
            logger.record_tabular("ExplainedVariance", special.explained_variance_1d(vpreds[:-1].ravel(), vtargs.ravel()))
            logger.record_tabular("Timesteps", total_timesteps)
            logger.record_tabular("TotalTimesteps", total_timesteps * batch_size)
            logger.record_tabular("TotalEpisodes", total_episodes)
            logger.record_tabular("MeanPredVal", vpreds.mean())

            logger.dump_tabular(**LOG_SETTINGS)

def _linesearch(f, x, fullstep, expected_improve_rate, max_backtracks=10, accept_ratio=.1):
    """
    Backtracking linesearch, where expected_improve_rate is the slope dy/dx at the initial point
    """
    fval = f(x)
    _log("fval before: %g"%fval)
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + stepfrac*fullstep
        newfval = f(xnew)
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate*stepfrac
        ratio = actual_improve/expected_improve
        _log("a/e/r: %g  /  %g  /  %g"%(actual_improve, expected_improve, ratio))
        if ratio > accept_ratio and actual_improve > 0:
            _log("fval after: %g"%newfval)
            return True, xnew
    return False, x

class LinearBaseline(object):
    def __init__(self, degree=1):
        self.coef_ = None
        self.intercept_ = None
        self.degree = degree
    def fit(self, X, y):
        if self.coef_ is None:
            self.coef_ = np.zeros(X.shape[1])
            self.intercept_ = 0

        assert X.ndim == 2 and y.ndim == 1
        feats = self.preproc(X)        
        residual = y - self.predict1(feats)

        from sklearn.linear_model import RidgeCV
        newmodel = RidgeCV()
        newmodel.fit(feats, residual)

        if np.isfinite(self.coef_).all():
            self.coef_ += newmodel.coef_
            self.intercept_ += newmodel.intercept_
        else:
            import IPython; IPython.embed(); asdf

    def predict(self, X):
        assert X.ndim == 2
        if self.coef_ is None:
            return np.zeros(len(X))
        else:
            return self.predict1(self.preproc(X))
    def predict1(self, feats):
        return feats.dot(self.coef_) + self.intercept_
    def preproc(self, X):
        return np.concatenate(compute_legendre_feats(X, self.degree), 1)

def compute_legendre_feats(X,degree):
    polynomials = [
        lambda x:x,
        lambda x:1.5*x**2-.5,
        lambda x:2.5*x**3 - 1.5*x,
        lambda x:4.375*x**4 - 3.75*x**2+.375]
    assert 0 <= degree <= 4
    return [p(X) for p in polynomials[:degree]]
