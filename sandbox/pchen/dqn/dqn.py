from sandbox.pchen.dqn.utils.common import *
from lasagne import init, nonlinearities

args = parser.parse_args()

N = 1
batch_size = args.batch_size
replay_min = args.replay_min  # 50000
network_update_freq = args.network_update_freq  # 10000
evaluation_freq = args.evaluation_freq  # 50000
evaluation_len = args.evaluation_len
exp_name = args.exp_name  # "test_dqn"
out_dir = args.out_dir  # "./"
checkpoint_freq = args.checkpoint_freq
min_eps = args.min_eps
algo = args.algo
temporal_frames = args.temporal_frames
dup_factor = args.dup_factor
validation_batch_size = args.validation_batch_size
shuffle_update_order = args.shuffle_update_order
load_expert_params = args.load_expert_params
select_max_by_current_q = args.select_max_by_current_q
regress_q_pi = args.regress_q_pi

rom_path = "vendor/atari_roms/%s.bin" % args.rom
mdp = AtariMDP(rom_path, obs_type=1, terminate_per_life=args.terminate_per_life)
mdp_test = AtariMDP(rom_path, obs_type=1)
gamma = args.gamma

action_set = mdp.action_set
reward_set = [0, 1]

network_input_img_size = (84, 84)
network_input_dim = (temporal_frames,) + network_input_img_size
network_output_dim = len(mdp.action_set)
reward_dim = len(reward_set)


def preprocess(rgb):
    # note here we follow the code that's released with Nature article instead of the one described in Minh 2013
    # to y
    y = rgb2y(rgb)
    # downsample (bilinear)
    # note imresize rependes on PIL https://github.com/python-pillow/Pillow
    sampled = scipy.misc.imresize(y, network_input_img_size, interp='bilinear')
    normalized = (sampled - np.mean(sampled)) / 128
    return normalized

# network consturction
if args.network == None:
    Q_X = T.tensor4("X")
    dqn_in = L.InputLayer(shape=(None,) + network_input_dim, input_var=Q_X)
    if args.network_args == "nature":
        dqn_conv1 = L.Conv2DLayer(dqn_in, 32, 8, stride=4, convolution=wrapped_conv)
        dqn_conv2 = L.Conv2DLayer(dqn_conv1, 64, 4, stride=2, convolution=wrapped_conv)
        dqn_conv3 = L.Conv2DLayer(dqn_conv2, 64, 3, stride=1, convolution=wrapped_conv)
        dqn_fc1 = L.DenseLayer(dqn_conv3, 512, nonlinearity=lasagne.nonlinearities.rectify)
        dqn_out = L.DenseLayer(dqn_fc1, network_output_dim, nonlinearity=lasagne.nonlinearities.identity)
    else:
        dqn_conv1 = L.Conv2DLayer(dqn_in, 16, 8, stride=4, convolution=wrapped_conv)
        dqn_conv2 = L.Conv2DLayer(dqn_conv1, 32, 4, stride=2, convolution=wrapped_conv)
        dqn_fc1 = L.DenseLayer(dqn_conv2, 256, nonlinearity=lasagne.nonlinearities.rectify)
        dqn_out = L.DenseLayer(dqn_fc1, network_output_dim, nonlinearity=lasagne.nonlinearities.identity)

    Q_vals = L.get_output(dqn_out)
    Q_args = [Q_X]
    Q_vals_fn = theano.function(Q_args, Q_vals)

    a_inds = T.ivector("A_inds")
    Q_tgt_vals = T.vector("Q_tgt_vals")
    Q_selected_vals = Q_vals[T.arange(Q_X.shape[0]), a_inds]

    Q_loss_args = Q_args + [a_inds, Q_tgt_vals]
    Q_loss = T.sum(T.square(Q_selected_vals - Q_tgt_vals)) / Q_X.shape[0]
    Q_loss_fn = theano.function(Q_loss_args, Q_loss)

    Q_params = L.get_all_params(dqn_out)
    init_from_args(Q_params, args)

    Q_updates = gen_updates(Q_loss, Q_params, args)

    Q_train_function = theano.function(Q_loss_args, Q_loss, updates=Q_updates)
elif args.network == "hier_platonic":
    att0_filters, att1_filters, use_softmax, rescale, use_final = \
        map(int, (args.network_args or "4,4,1,3,1").split(","))
    use_softmax = bool(use_softmax)
    rescale = float(rescale)
    use_final = bool(use_final)
    Q_X = T.tensor4("X")
    dqn_in = L.InputLayer(shape=(batch_size, ) + network_input_dim, input_var=Q_X)
    dqn_conv0_attention = L.Conv2DLayer(dqn_in, 32, 8, stride=4, convolution=wrapped_conv)
    dqn_conv0_vals = L.DenseLayer(
        L.DenseLayer(PlatonicLayer(dqn_in, dqn_conv0_attention, softmaxed=use_softmax), 64),
        network_output_dim, nonlinearity=nonlinearities.identity)
    dqn_conv1 = L.Conv2DLayer(dqn_in, 16, 8, stride=4, convolution=wrapped_conv)
    dqn_conv1_attention = L.Conv2DLayer(dqn_conv1, 2, 4, stride=2, convolution=wrapped_conv)
    dqn_conv1_vals = L.DenseLayer(
        L.DenseLayer(PlatonicLayer(dqn_conv1, dqn_conv1_attention, softmaxed=use_softmax), 64),
        network_output_dim, nonlinearity=nonlinearities.identity)
    dqn_conv2 = L.Conv2DLayer(dqn_conv1, 32, 4, stride=2, convolution=wrapped_conv)
    dqn_fc1 = L.DenseLayer(dqn_conv2, 256, nonlinearity=lasagne.nonlinearities.rectify)
    dqn_out = L.DenseLayer(dqn_fc1, network_output_dim, nonlinearity=lasagne.nonlinearities.identity)

    Q_vals = L.get_output(dqn_out) * float(use_final)
    Q_vals = (Q_vals + L.get_output(dqn_conv0_vals) + L.get_output(dqn_conv1_vals)) / rescale
    Q_args = [Q_X]
    Q_vals_fn_orig = theano.function(Q_args, Q_vals)
    # we actually have to make row size = batch_size
    def Q_vals_fn(qx):
        bs = qx.shape[0]
        if bs > batch_size:
            assert "not implemented"
        if bs < batch_size:
            wrapped_qx = np.zeros((batch_size,) + network_input_dim, dtype='float32')
            wrapped_qx[np.arange(bs)] = qx
        else:
            wrapped_qx = qx
        return Q_vals_fn_orig(wrapped_qx)[:bs]

    a_inds = T.ivector("A_inds")
    Q_tgt_vals = T.vector("Q_tgt_vals")
    Q_selected_vals = Q_vals[T.arange(Q_X.shape[0]), a_inds]

    Q_loss_args = Q_args + [a_inds, Q_tgt_vals]
    Q_loss = T.sum(T.square(Q_selected_vals - Q_tgt_vals)) / Q_X.shape[0]

    Q_params = L.get_all_params(dqn_out)
    init_from_args(Q_loss_args, Q_params)
    Q_updates = gen_updates(Q_loss, Q_params, args)
    Q_train_function = theano.function(Q_loss_args, Q_loss, updates=Q_updates)
elif args.network == "additive_qvals":
    pass
    # Q_X = T.tensor4("X")
    # dqn_in = L.InputLayer(shape=(batch_size, ) + network_input_dim, input_var=Q_X)
    # dqn_conv0_vals =
    # dqn_conv1 = L.Conv2DLayer(dqn_in, 16, 8, stride=4, convolution=wrapped_conv)
    # dqn_conv1_attention = L.Conv2DLayer(dqn_conv1, 2, 4, stride=2, convolution=wrapped_conv)
    # dqn_conv1_vals = L.DenseLayer(
    #     L.DenseLayer(PlatonicLayer(dqn_conv1, dqn_conv1_attention), 64),
    #     network_output_dim, nonlinearity=nonlinearities.identity)
    # dqn_conv2 = L.Conv2DLayer(dqn_conv1, 32, 4, stride=2, convolution=wrapped_conv)
    # dqn_fc1 = L.DenseLayer(dqn_conv2, 256, nonlinearity=lasagne.nonlinearities.rectify)
    # dqn_out = L.DenseLayer(dqn_fc1, network_output_dim, nonlinearity=lasagne.nonlinearities.identity)
    #
    # Q_vals = (L.get_output(dqn_out) + L.get_output(dqn_conv0_vals) + L.get_output(dqn_conv1_vals)) / 3.
    # Q_args = [Q_X]
    # Q_vals_fn_orig = theano.function(Q_args, Q_vals)
    # # we actually have to make row size = batch_size
    # def Q_vals_fn(qx):
    #     bs = qx.shape[0]
    #     if bs > batch_size:
    #         assert "not implemented"
    #     if bs < batch_size:
    #         wrapped_qx = np.zeros((batch_size,) + network_input_dim, dtype='float32')
    #         wrapped_qx[np.arange(bs)] = qx
    #     else:
    #         wrapped_qx = qx
    #     return Q_vals_fn_orig(wrapped_qx)[:bs]
    #
    # a_inds = T.ivector("A_inds")
    # Q_tgt_vals = T.vector("Q_tgt_vals")
    # Q_selected_vals = Q_vals[T.arange(Q_X.shape[0]), a_inds]
    #
    # Q_loss_args = Q_args + [a_inds, Q_tgt_vals]
    # Q_loss = T.sum(T.square(Q_selected_vals - Q_tgt_vals)) / Q_X.shape[0]
    #
    # Q_params = L.get_all_params(dqn_out)
    # Q_updates = gen_updates(Q_loss, Q_params, args)
    # Q_train_function = theano.function(Q_loss_args, Q_loss, updates=Q_updates)
else:
    raise "unsupported network"

if algo == "double":
    alternative_param_vals = copy_from_params(Q_params)
    def init_like(val):
        shape = val.shape
        if len(shape) >= 2:
            return lasagne.init.GlorotNormal().sample(shape)
        else:
            return lasagne.init.Uniform().sample(shape)
    alternative_param_vals = map(init_like, alternative_param_vals)
    prev_alternative_param_vals = alternative_param_vals

if load_expert_params:
    now_Q_param_vals = copy_from_params(Q_params)
    load_vars(Q_params, load_expert_params)
    expert_Q_param_vals = copy_from_params(Q_params)
    set_to_params(now_Q_param_vals, Q_params)

# memory :: Matrix StepId (Observation, Action, Terminal, Reward)
memory_size = 190000
observation_mask = slice(0, network_input_img_size[0] * network_input_img_size[1])
action_mask = slice(observation_mask.stop, observation_mask.stop + network_output_dim)
terminal_mask = slice(action_mask.stop, action_mask.stop + 1)
reward_mask = slice(terminal_mask.stop, terminal_mask.stop + 1)

memory = np.zeros((memory_size, reward_mask.stop), dtype='float32')


def ind_from_action(action):
    return list(action_set).index(action)


def vec_from_action(action):
    ind = ind_from_action(action)
    vec = np.zeros(network_output_dim)
    vec[ind] = 1
    return vec


def sample_actions(N):
    return [random.choice(action_set) for _ in xrange(N)]


def greedy_actions(q_func, obs):
    q_vals = q_func(obs)
    a_inds = np.argmax(q_vals, axis=1)
    return [mdp.action_set[i] for i in a_inds]


def convert_reward(r):
    out = np.zeros((r.shape[0], reward_dim))
    out[np.arange(r.shape[0]), np.asarray(r > 0, dtype='int32')] = 1
    out[np.arange(r.shape[0]), np.asarray(r < 0, dtype='int32')] = -1
    return out

def prepare_inputX(memory, idx):
    bs = len(idx)
    bnX = np.zeros((bs,) + network_input_dim, dtype='float32')
    for i, ind in zip(xrange(bs), idx):
        bnX[i] = memory[(ind - temporal_frames + 1):(ind + 1), observation_mask].reshape(network_input_dim)
    return bnX

def update_from_memory_batch(q_func, updater, memory, batch_inds, gamma, prev_param_vals):
    if algo == "double":
        if load_expert_params:
            raise "not implement"
        global alternative_param_vals, prev_alternative_param_vals
        bs = len(batch_inds)
        bnX = prepare_inputX(memory, batch_inds + 1)
        staged = copy_from_params(Q_params)
        set_to_params(prev_param_vals, Q_params)
        this_q_vals = q_func(bnX)
        set_to_params(prev_alternative_param_vals, Q_params)
        other_q_vals = q_func(bnX)
        this_q_max_idx = np.argmax(this_q_vals, axis=1)
        other_q_max_idx = np.argmax(other_q_vals, axis=1)
        this_selected_next_q = this_q_vals[np.arange(bs), other_q_max_idx]
        other_selected_next_q = other_q_vals[np.arange(bs), this_q_max_idx]
        this_ys = memory[batch_inds, reward_mask].reshape(-1) + \
             (gamma * this_selected_next_q) \
             * (1 - memory[batch_inds, terminal_mask]).reshape(-1)
        other_ys = memory[batch_inds, reward_mask].reshape(-1) + \
                  (gamma * other_selected_next_q) \
                  * (1 - memory[batch_inds, terminal_mask]).reshape(-1)

        bnX = prepare_inputX(memory, batch_inds)
        set_to_params(alternative_param_vals, Q_params)
        other_diff = updater(bnX, ind_from_vecs(memory[batch_inds, action_mask]).astype('int32'), other_ys)
        alternative_param_vals = copy_from_params(Q_params)
        set_to_params(staged, Q_params)
        this_diff = updater(bnX, ind_from_vecs(memory[batch_inds, action_mask]).astype('int32'), this_ys)
        return this_diff, other_diff
    else:
        bs = len(batch_inds)
        bnX = prepare_inputX(memory, batch_inds + 1)
        staged = copy_from_params(Q_params)
        if load_expert_params and not select_max_by_current_q:
            set_to_params(expert_Q_param_vals, Q_params)
        else:
            set_to_params(prev_param_vals, Q_params)

        if not regress_q_pi:
            q_max_idx = ind_from_vecs(memory[batch_inds, action_mask]).astype('int32')
        else:
            q_max_idx = np.argmax(Q_vals_fn(bnX), axis=1)

        if load_expert_params and select_max_by_current_q:
            set_to_params(expert_Q_param_vals, Q_params)

        ys = memory[batch_inds, reward_mask].reshape(-1) + \
             (gamma * (q_func(bnX))[np.arange(bs), q_max_idx]) \
             * (1 - memory[batch_inds, terminal_mask]).reshape(-1)
        # reuse to save meaningless memory
        bnX = prepare_inputX(memory, batch_inds)
        set_to_params(staged, Q_params)
        return updater(bnX, ind_from_vecs(memory[batch_inds, action_mask]).astype('int32'), ys)



def evaluate(max_len=evaluation_len):
    temp_memory = np.zeros((max_len, reward_mask.start + 1), dtype='float32')

    period_plays = 0
    temp_memory_i = 0
    play_reward = 0
    play_rewards = []

    states, obs = mdp_test.sample_initial_states(N)
    for step_i in xrange(1, 10000000000):
        temp_memory[temp_memory_i, observation_mask] = preprocess(obs[0]).ravel()
        if temp_memory_i < (temporal_frames - 1) or random.random() <= 0.05:
            actions = sample_actions(1)
        else:
            bX = temp_memory[(temp_memory_i - temporal_frames + 1):(temp_memory_i + 1), observation_mask].reshape(
                (1,) + network_input_dim)
            actions = greedy_actions(Q_vals_fn, bX)

        next_states, next_obs, rewards_n, done_n = mdp_test.step(states, actions, span=4)
        for i, action, reward, done, ob in zip(*([xrange(N), actions, rewards_n, done_n, obs])):
            play_reward += reward
            if done:
                period_plays += 1
                play_rewards.append(play_reward)
                play_reward = 0

        states, obs = next_states, next_obs
        temp_memory_i += 1

        if (step_i % max_len) == 0:
            length = max_len / (float(period_plays) if period_plays != 0 else 1)
            reward = np.mean(play_rewards)
            std = np.std(play_rewards)
            return (length, reward, std)


memory_i = 0
t = time.time()
period_i = 0
period_plays = 0
period_rewards = 0
to_train_inds = []

states, obs = mdp.sample_initial_states(N)


if __name__ == '__main__':
    for step_i, this_stat in experiment_iter(
            args.max_iter, out_dir, exp_name, checkpoint_freq, Q_params):

        memory[memory_i, observation_mask] = preprocess(obs[0]).ravel()

        eps = max(min_eps, 1 - 0.9 / 1e6 * step_i)
        if np.random.rand() < eps or memory_i < (temporal_frames-1):
            actions = sample_actions(N)
        else:
            bX = memory[(memory_i - temporal_frames + 1):(memory_i + 1), observation_mask].reshape(
                (1,) + network_input_dim)
            actions = greedy_actions(Q_vals_fn, bX)

        next_states, next_obs, rewards_n, done_n = mdp.step(states, actions, span=4)
        for i, action, reward, done, ob in zip(*([xrange(N), actions, rewards_n, done_n, obs])):
            memory[memory_i, action_mask] = vec_from_action(actions[0])
            memory[memory_i, terminal_mask] = int(done)
            memory[memory_i, reward_mask] = 0 if reward == 0 else (1 if reward > 0 else -1)
            memory_i = (memory_i + 1) % memory_size
            if done:
                period_plays += 1
            period_rewards += reward
        states, obs = next_states, next_obs

        if step_i >= replay_min:
            for _ in xrange(dup_factor):
                to_train_inds.append(np.random.randint(temporal_frames - 1, min(step_i, memory_size - 1) - 1, batch_size))
            if (step_i % network_update_freq) == 0:
                this_stat['sampling_time'] = time.time() - t
                t = time.time()
                prev_param_vals = copy_from_params(Q_params)
                errors = []
                validation_batch_inds = np.random.randint(temporal_frames - 1, min(step_i, memory_size - 1) - 1, validation_batch_size)
                validation_bnX = prepare_inputX(memory, validation_batch_inds)
                validation_prev_qvals = Q_vals_fn(validation_bnX)
                if algo == "double":
                    prev_alternative_param_vals = copy_vals(alternative_param_vals)
                    other_errors = []
                if shuffle_update_order:
                    random.shuffle(to_train_inds)
                validation_errs = []
                for train_i, sample_inds in enumerate(to_train_inds):
                    cost = update_from_memory_batch(Q_vals_fn, Q_train_function, memory, sample_inds,
                                                    gamma, prev_param_vals)
                    if algo == "double":
                        cost, other_cost = cost
                        other_errors.append(other_cost)
                    errors.append(cost)
                    if train_i % (len(to_train_inds) / 10) == 0:
                        validation_cur_qvals = Q_vals_fn(validation_bnX)
                        validation_errs.append(np.mean(np.square(validation_prev_qvals - validation_cur_qvals)))

                this_stat['q_err_mean'] = np.mean(errors)
                this_stat['q_err_std'] = np.std(errors)
                if algo == "double":
                    this_stat['other_q_err_mean'] = np.mean(other_errors)
                    this_stat['other_q_err_std'] = np.std(other_errors)
                validation_cur_qvals = Q_vals_fn(validation_bnX)
                this_stat['validation_errs'] = validation_errs
                this_stat['validation_q_avg'] = np.mean(validation_cur_qvals)
                this_stat['validation_diff'] = np.mean(np.square(validation_prev_qvals - validation_cur_qvals))
                this_stat['optimization_time'] = time.time() - t
                f_safe_period_plays = float(max(period_plays, 1))
                this_stat['train_len'] = network_update_freq / f_safe_period_plays
                this_stat['train_reward_avg'] = period_rewards / f_safe_period_plays
                period_i += 1
                period_plays = 0
                period_rewards = 0
                to_train_inds = []
                t = time.time()

            if (step_i % evaluation_freq) == 0:
                this_stat['eval_len'], this_stat['eval_reward_mean'], \
                this_stat['eval_reward_std'] = evaluate()
