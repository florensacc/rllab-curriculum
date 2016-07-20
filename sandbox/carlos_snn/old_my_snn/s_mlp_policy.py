import lasagne
import lasagne.layers as L
import lasagne.nonlinearities as NL
import theano.tensor as TT
import numpy as np
from contextlib import contextmanager

from rllab.core.lasagne_layers import ParamLayer
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.network import MLP
from rllab.spaces import Box

from rllab.sampler.utils import rollout  # I need this for logging the diagnostics: run the policy with all diff latents

from rllab.core.serializable import Serializable
from rllab.policies.base import StochasticPolicy
from rllab.misc.overrides import overrides
from rllab.misc import logger
from rllab.misc import ext
from rllab.misc import autoargs
from rllab.distributions.diagonal_gaussian import DiagonalGaussian

from sandbox.carlos_snn.distributions.categorical import Categorical
from sandbox.rocky.snn.distributions.bernoulli import Bernoulli


class GaussianMLPPolicy_snn(StochasticPolicy, LasagnePowered, Serializable):
    @autoargs.arg('hidden_sizes', type=int, nargs='*',
                  help='list of sizes for the fully-connected hidden layers')
    @autoargs.arg('std_sizes', type=int, nargs='*',
                  help='list of sizes for the fully-connected layers for std, note'
                       'there is a difference in semantics than above: here an empty'
                       'list means that std is independent of input and the last size is ignored')
    @autoargs.arg('initial_std', type=float,
                  help='Initial std')
    @autoargs.arg('std_trainable', type=bool,
                  help='Is std trainable')
    @autoargs.arg('output_nl', type=str,
                  help='nonlinearity for the output layer')
    @autoargs.arg('nonlinearity', type=str,
                  help='nonlinearity used for each hidden layer, can be one '
                       'of tanh, sigmoid')
    @autoargs.arg('bn', type=bool,
                  help='whether to apply batch normalization to hidden layers')
    def __init__(
            self,
            env_spec,
            ##CF - latents units at the input
            latent_dim=2,
            latent_name='bernoulli',
            bilinear_integration=False,
            resample=False,
            hidden_sizes=(32, 32),
            learn_std=True,
            init_std=1.0,
            adaptive_std=False,
            std_share_network=False,
            std_hidden_sizes=(32, 32),
            std_hidden_nonlinearity=NL.tanh,
            hidden_nonlinearity=NL.tanh,
            output_nonlinearity=None,
            min_std=1e-4,
    ):
        self.latent_dim = latent_dim  ##could I avoid needing this self for the get_action?
        self.latent_name = latent_name
        self.bilinear_integration = bilinear_integration
        self.resample = resample
        self.min_std = min_std

        self.pre_fix_latent = np.array([])  # if this is not empty when using reset() it will use this latent
        self.latent_fix = np.array([])  # this will hold the latents variable sampled in reset()
        self._set_std_to_0 = False

        if latent_name == 'normal':
            self.latent_dist = DiagonalGaussian(self.latent_dim)
            self.latent_dist_info = dict(mean=np.zeros(self.latent_dim), log_std=np.zeros(self.latent_dim))
        elif latent_name == 'bernoulli':
            self.latent_dist = Bernoulli(self.latent_dim)
            self.latent_dist_info = dict(p=0.5 * np.ones(self.latent_dim))
        elif latent_name == 'categorical':
            self.latent_dist = Categorical(self.latent_dim)
            if self.latent_dim > 0:
                self.latent_dist_info = dict(prob=1./self.latent_dim * np.ones(self.latent_dim))
            else:
                self.latent_dist_info = dict(prob=np.ones(self.latent_dim))
            print self.latent_dist_info
        else:
            raise NotImplementedError

        Serializable.quick_init(self, locals())
        assert isinstance(env_spec.action_space, Box)

        ##CF - enlarge obs with the latents
        if self.bilinear_integration:
            obs_dim = env_spec.observation_space.flat_dim + latent_dim +\
                      env_spec.observation_space.flat_dim * latent_dim
        else:
            obs_dim = env_spec.observation_space.flat_dim + latent_dim  # here only if concat.

        action_dim = env_spec.action_space.flat_dim

        # create network
        mean_network = MLP(
            input_shape=(obs_dim,),
            output_dim=action_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=output_nonlinearity,
        )

        l_mean = mean_network.output_layer
        obs_var = mean_network.input_layer.input_var

        if adaptive_std:
            l_log_std = MLP(
                input_shape=(obs_dim,),
                input_var=obs_var,
                output_dim=action_dim,
                hidden_sizes=std_hidden_sizes,
                hidden_nonlinearity=std_hidden_nonlinearity,
                output_nonlinearity=None,
            ).output_layer
        else:
            l_log_std = ParamLayer(
                mean_network.input_layer,
                num_units=action_dim,
                param=lasagne.init.Constant(np.log(init_std)),
                name="output_log_std",
                trainable=learn_std,
            )

        mean_var, log_std_var = L.get_output([l_mean, l_log_std])

        if self.min_std is not None:
            log_std_var = TT.maximum(log_std_var, np.log(self.min_std))

        self._l_mean = l_mean
        self._l_log_std = l_log_std

        self._dist = DiagonalGaussian(action_dim)

        LasagnePowered.__init__(self, [l_mean, l_log_std])
        super(GaussianMLPPolicy_snn, self).__init__(env_spec)

        self._f_dist = ext.compile_function(
            inputs=[obs_var],
            outputs=[mean_var, log_std_var],
        )

#  this is currently not used, although it could, in dist_info_sym and in get_actions. Also we could refactor all..
        latent_var = Box(low=-np.inf, high=np.inf, shape=(1,)).new_tensor_variable('latents', extra_dims=1)
        extended_obs_var = TT.concatenate([obs_var, latent_var,
                                           TT.flatten(obs_var[:, :, np.newaxis] * latent_var[:, np.newaxis, :],
                                                      outdim=2)]
                                          , axis=1)
        self._extended_obs_var = ext.compile_function(
            inputs=[obs_var, latent_var],
            outputs=[extended_obs_var]
        )
    ##CF
    @property
    def latent_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(1,))

    ##

    ##CF - the mean and var now also depend on the particular latents sampled

    def dist_info_sym(self, obs_var, latent_var):
        # generate the generalized input (append latents to obs.)
        if self.bilinear_integration:
            extended_obs_var = TT.concatenate([obs_var, latent_var,
                                               TT.flatten(obs_var[:, :, np.newaxis] * latent_var[:, np.newaxis, :],
                                                          outdim=2)]
                                              , axis=1)
        else:
            extended_obs_var = TT.concatenate([obs_var, latent_var], axis=1)
        mean_var, log_std_var = L.get_output([self._l_mean, self._l_log_std], extended_obs_var)
        if self.min_std is not None:
            log_std_var = TT.maximum(log_std_var, np.log(self.min_std))
        return dict(mean=mean_var, log_std=log_std_var)

    @overrides
    def get_action(self, observation):
        actions, outputs = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in outputs.iteritems()}

    def get_actions(self, observations):
        ##CF
        # how can I impose that I only reset for a whole rollout? before calling get_actions!!
        observations = np.array(observations)  # needed to do the outer product for the bilinear
        if self.latent_dim:
            if self.resample:
                latents = [self.latent_dist.sample(self.latent_dist_info) for _ in observations]
                # print 'resampling the latents'
            else:
                if not np.size(self.latent_fix) == self.latent_dim:  # we decide to reset based on if smthing in the fix
                    # logger.log('Reset for latents: the latent_fix {} not match latent_dim{}'.format(self.latent_fix, self.latent_dim))
                    self.reset()
                if len(self.pre_fix_latent) == self.latent_dim:  # If we have a pre_fix, reset will put the latent to it
                    # logger.log('Reset for latents: we have a pre_fix to fix!')
                    self.reset()  # this overwrites the latent sampled or in latent_fix
                latents = np.tile(self.latent_fix, [len(observations), 1])  # maybe a broadcast operation better...
                # print 'not resample, use latent_fix, obtaining: ', latents
            if self.bilinear_integration:
                # print 'the obs is: ' , observations, '\nwith time length: {}\n'.format(observations.shape[0])
                # print 'the reshaped bilinear is:\n' , np.reshape(observations[:, np.newaxis, :] * latents[:, :, np.newaxis],
                #                                    (observations.shape[0], -1) )
                extended_obs = np.concatenate([observations, latents,
                                               np.reshape(
                                                   observations[:, :, np.newaxis] * latents[:, np.newaxis, :],
                                                   (observations.shape[0], -1))],
                                              axis=1)
                # print 'Latents: {}, observations: {}'.format(latents, observations), \
                #     'The extended obs are: ', extended_obs, \
                #     '\ndone with the theano function it is', self._extended_obs_var(observations,latents)
            else:
                extended_obs = np.concatenate([observations, latents], axis=1)
        else:
            latents = np.array([[]] * len(observations))
            extended_obs = observations
        # print 'the extened_obs are:\n', extended_obs
        # make mean, log_std also depend on the latents (as observ.)
        mean, log_std = self._f_dist(extended_obs)

        if self._set_std_to_0:
            actions = mean
            log_std = -1e6 * np.ones_like(log_std)
        else:
            rnd = np.random.normal(size=mean.shape)
            actions = rnd * np.exp(log_std) + mean
        # print latents
        return actions, dict(mean=mean, log_std=log_std, latents=latents)

    def set_pre_fix_latent(self, latent):
        self.pre_fix_latent = np.array(latent)

    def unset_pre_fix_latent(self):
        self.pre_fix_latent = np.array([])

    @contextmanager
    def set_std_to_0(self):
        print 'about to set the std to 0'
        self._set_std_to_0 = True
        # try:
        #     print 'now doig something'
        #     yield
        # finally:
        #     print 'and finally closing it'
        #     self._set_std_to_0 = False
        yield
        print 'all the work done!'
        self._set_std_to_0 = False

    def unset_std_to_0(self):
        self._set_std_to_0 = False

    @overrides
    def reset(self):  # executed at the start of every rollout. Will fix the latent if needed.
        # print 'entering reset'
        if not self.resample:
            if self.pre_fix_latent.size > 0:
                self.latent_fix = self.pre_fix_latent
            else:
                self.latent_fix = self.latent_dist.sample(self.latent_dist_info)
                # print 'I reset to latent {} because the pre_fix_latent is {}'.format(self.latent_fix, self.pre_fix_latent)
        else:
            pass

    def log_diagnostics(self, paths):
        log_stds = np.vstack([path["agent_infos"]["log_std"] for path in paths])
        logger.record_tabular('MaxPolicyStd', np.max(np.exp(log_stds)))
        logger.record_tabular('MinPolicyStd', np.min(np.exp(log_stds)))
        logger.record_tabular('AveragePolicyStd', np.mean(np.exp(log_stds)))

    @property
    def distribution(self):
        """
        We set the distribution to the policy itself since we need some behavior different from a usual diagonal
        Gaussian distribution.
        """
        return self._dist

    def log_likelihood(self, actions, agent_infos, action_only=True):
        # First compute logli of the action. This assumes the latents FIX to whatever was sampled, and hence we only
        # need to use the mean and log_std, but not any information about the latents
        logli = self._dist.log_likelihood(actions, agent_infos)
        if not action_only:
            raise NotImplementedError
            #   if not action_only:
            #       for idx, latent_name in enumerate(self._latent_distributions):
            #           latent_var = dist_info["latent_%d" % idx]
            #           prefix = "latent_%d_" % idx
            #           latent_dist_info = {k[len(prefix):]: v for k, v in dist_info.iteritems() if k.startswith(
            #               prefix)}
            #           logli += latent_name.log_likelihood(latent_var, latent_dist_info)
        return logli
