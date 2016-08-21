from sandbox.rein.sampler.utils import rollout
from rllab.sampler.stateful_pool import singleton_pool
from rllab.misc import ext
from rllab.misc import logger
from rllab.misc import tensor_utils
import numpy as np


def _worker_init(G, id):
    if singleton_pool.n_parallel > 1:
        import os
        os.environ['THEANO_FLAGS'] = 'device=cpu,floatX=float32'
    G.worker_id = id


def initialize(n_parallel):
    singleton_pool.initialize(n_parallel)
    singleton_pool.run_each(
        _worker_init, [(id,) for id in xrange(singleton_pool.n_parallel)])


def _worker_populate_task(G, env, policy, dynamics):
    G.env = env
    G.policy = policy
    G.dynamics = dynamics


def populate_task(env, policy, dynamics):
    logger.log("Populating workers...")
    singleton_pool.run_each(
        _worker_populate_task,
        [(env, policy, dynamics)] * singleton_pool.n_parallel
    )
    logger.log("Populated")


def _worker_set_seed(_, seed):
    ext.set_seed(seed)


def set_seed(seed):
    singleton_pool.run_each(
        _worker_set_seed,
        [(seed + i,) for i in xrange(singleton_pool.n_parallel)]
    )


def _worker_set_policy_params(G, params):
    G.policy.set_param_values(params)


def _worker_set_dynamics_params(G, params):
    G.dynamics.set_param_values(params)


def _worker_collect_one_path(G, max_path_length, itr, normalize_reward,
                             reward_mean, reward_std, kl_batch_size, n_itr_update, use_replay_pool,
                             obs_mean, obs_std, act_mean, act_std, second_order_update, predict_reward, surprise_type,
                             num_seq_frames):
    # Path rollout.
    path = rollout(G.env, G.policy, max_path_length, num_seq_frames=num_seq_frames)

    # Save original reward.
    path['rewards_orig'] = np.array(path['rewards'])

    # We skip first iteration as it is often difficult to normalize the KL
    # divergence terms.
    if itr > -1:
        # Iterate over all paths and compute intrinsic reward by updating the
        # model on each observation, calculating the KL divergence of the new
        # params to the old ones, and undoing this operation.
        obs = (path['observations'] * G.dynamics.num_classes).astype(int)
        single_obs_dim = obs.shape[1] / num_seq_frames
        act = path['actions']
        rew_orig = path['rewards_orig']
        # inputs = (o,a), target = o'
        obs_nxt = np.vstack((obs[1:, -single_obs_dim:]))
        # Making sure everything aligns for num_seq_framesd
        # _a = obs[:-1]
        # _b = np.concatenate((np.zeros((3, obs.shape[1])), _a), axis=0)
        # _c = np.array([_b[num_seq_frames - i - 1:-i] for i in (np.array(range(num_seq_frames - 1)) + 1)])
        # _e = np.transpose(np.concatenate((_a[np.newaxis, :, :], _c), axis=0), (1, 0, 2))
        # _f = _e.reshape((_e.shape[0], -1))
        _inputs = np.hstack([obs[:-1], act[:-1]])
        _targets = obs_nxt
        # FIXME: turned on by default
        predict_reward = True
        if predict_reward:
            _targets = np.hstack((_targets, rew_orig[:-1, None]))

        # KL vector assumes same shape as reward.
        kl = np.zeros(rew_orig.shape)

        for j in xrange(int(np.ceil((obs.shape[0] - 1) / float(kl_batch_size)))):

            start = j * kl_batch_size
            end = np.minimum((j + 1) * kl_batch_size, _inputs.shape[0])

            if surprise_type == G.dynamics.SurpriseType.INFGAIN:
                if second_order_update:
                    G.dynamics.save_params()
                    step_size = 1.0
                    surpr = G.dynamics.train_update_fn(
                        _inputs[start:end], _targets[start:end], step_size)
                elif use_replay_pool:
                    G.dynamics.save_params()
                    for _ in xrange(n_itr_update):
                        G.dynamics.train_update_fn(
                            _inputs[start:end], _targets[start:end], 1.0)
                    surpr = G.dynamics.fn_kl()
                    G.dynamics.load_prev_params()
                else:
                    surpr = np.nan

            elif surprise_type == G.dynamics.SurpriseType.BALD:
                surpr = G.dynamics.train_update_fn(_inputs[start:end])

            elif surprise_type == G.dynamics.SurpriseType.VAR:
                surpr = G.dynamics.train_update_fn(_inputs[start:end])

            elif surprise_type == G.dynamics.SurpriseType.L1:
                assert use_replay_pool
                G.dynamics.save_params()
                for _ in xrange(n_itr_update):
                    G.dynamics.train_update_fn(
                        _inputs[start:end], _targets[start:end], 1.0)
                surpr = G.dynamics.fn_l1()
                G.dynamics.load_prev_params()

            elif surprise_type == G.dynamics.SurpriseType.COMPR:
                if second_order_update:
                    # G.dynamics.save_params()
                    #
                    # logp_before = G.dynamics.fn_logp(
                    #     _inputs[start:end], _targets[start:end])
                    # # conservative step (actual step should be 1.0)
                    # step_size = 1.0
                    # G.dynamics.train_update_fn(
                    #     _inputs[start:end], _targets[start:end], step_size)
                    # # Calculate current minibatch surprise.
                    # logp_after = G.dynamics.fn_logp(
                    #     _inputs[start:end], _targets[start:end])
                    # G.dynamics.load_prev_params()
                    #
                    # surpr = logp_after - logp_before
                    surpr = - G.dynamics.fn_logp(
                        _inputs[start:end], _targets[start:end])
                    surpr[surpr < 0] = 0.

                else:
                    surpr = np.nan

            # Load suprise into np.array.
            for k in xrange(start, end):
                if isinstance(surpr, float) or len(surpr.shape) == 0:
                    kl[k] = surpr
                else:
                    kl[k] = surpr[k - start]

        # Last element in KL vector needs to be replaced by second last one
        # because the actual last observation has no next observation.
        if len(path['rewards']) > 1:
            kl[-1] = kl[-2]

        # Stuff it in path
        path['KL'] = kl

    return path, len(path["rewards"])


def sample_paths(
        policy_params,
        dynamics_params,
        max_samples,
        max_path_length=np.inf,
        itr=None,
        normalize_reward=None,
        reward_mean=None,
        reward_std=None,
        kl_batch_size=None,
        n_itr_update=None,
        use_replay_pool=None,
        obs_mean=None,
        obs_std=None,
        act_mean=None,
        act_std=None,
        second_order_update=None,
        predict_reward=None,
        surprise_type=None,
        num_seq_frames=1
):
    """
    :param policy_params: parameters for the policy. This will be updated on each worker process
    :param max_samples: desired maximum number of samples to be collected. The actual number of collected samples
    might be greater since all trajectories will be rolled out either until termination or until max_path_length is
    reached
    :param max_path_length: horizon / maximum length of a single trajectory
    :return: a list of collected paths
    """
    singleton_pool.run_each(
        _worker_set_policy_params,
        [(policy_params,)] * singleton_pool.n_parallel
    )

    # Set dynamics params.
    # --------------------
    singleton_pool.run_each(
        _worker_set_dynamics_params,
        [(dynamics_params,)] * singleton_pool.n_parallel
    )
    # --------------------
    return singleton_pool.run_collect(
        _worker_collect_one_path,
        threshold=max_samples,
        args=(max_path_length,
              itr,
              normalize_reward,
              reward_mean,
              reward_std,
              kl_batch_size,
              n_itr_update,
              use_replay_pool,
              obs_mean,
              obs_std,
              act_mean,
              act_std,
              second_order_update,
              predict_reward,
              surprise_type,
              num_seq_frames),
        show_prog_bar=True
    )


def truncate_paths(paths, max_samples):
    """
    Truncate the list of paths so that the total number of samples is exactly equal to max_samples. This is done by
    removing extra paths at the end of the list, and make the last path shorter if necessary
    :param paths: a list of paths
    :param max_samples: the absolute maximum number of samples
    :return: a list of paths, truncated so that the number of samples adds up to max-samples
    """
    # chop samples collected by extra paths
    # make a copy
    paths = list(paths)
    total_n_samples = sum(len(path["rewards"]) for path in paths)
    while len(paths) > 0 and total_n_samples - len(paths[-1]["rewards"]) >= max_samples:
        total_n_samples -= len(paths.pop(-1)["rewards"])
    if len(paths) > 0:
        last_path = paths.pop(-1)
        truncated_last_path = dict()
        truncated_len = len(
            last_path["rewards"]) - (total_n_samples - max_samples)
        for k, v in last_path.iteritems():
            if k in ["observations", "actions", "rewards"]:
                truncated_last_path[k] = tensor_utils.truncate_tensor_list(
                    v, truncated_len)
            elif k in ["env_infos", "agent_infos"]:
                truncated_last_path[k] = tensor_utils.truncate_tensor_dict(
                    v, truncated_len)
            else:
                raise NotImplementedError
        paths.append(truncated_last_path)
    return paths
