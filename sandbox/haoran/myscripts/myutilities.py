import matplotlib.pyplot as plt
import numpy as np
import joblib
import time
import sys
import os
import csv

def collect_data(algo, batch_size=None, n_parallel=4):
    from rllab.sampler import parallel_sampler
    parallel_sampler.initialize(n_parallel=n_parallel)

    from rllab.algos.batch_polopt import BatchPolopt
    # assert isinstance(algo, BatchPolopt)
    ori_batch_size = algo.batch_size
    if batch_size is not None:
        algo.batch_size = batch_size
    algo.start_worker()
    algo.init_opt()
    itr = 0
    paths = algo.obtain_samples(itr,phase="train")
    all_data = algo.process_samples(itr, paths,phase="train")
    algo.shutdown_worker()
    algo.batch_size = ori_batch_size

    return all_data


def trpo_plot_by_batchsize(dir_groups,name_groups,stat,xlim=None,ylim=None,figsize=None,smoothing=None):
    # setup
    assert(len(dir_groups)==len(name_groups))
    if figsize is not None:
        fig,ax = plt.subplots(figsize=figsize)
    else:
        fig,ax = plt.subplots()
    hsv = plt.get_cmap('hsv')
    colors = hsv(np.linspace(0, 1.0, len(name_groups)+1))[:-1]

    # loop over different param settings
    for i,color,dirs in zip(list(range(len(colors))),colors,dir_groups):
        data = []
        if len(dirs) == 0:
            print("Emptry directory list %s"%(name_groups[i]))
            sys.exit(1)

        # loop over different seeds
        for directory in dirs:
            # read progress.csv
            import os
            progress_file = "%s/progress.csv"%(directory)
            if not os.path.isfile(progress_file):
                print("No progress.csv file for %s"%(directory))
                sys.exit(1)
            progress = read_csv(progress_file)

            # read the training curve
            datum = np.array(progress[stat])
            if smoothing is not None:
                weights = np.repeat(1.0, smoothing)/smoothing
                datum = np.convolve(datum,weights,mode='same')
            data.append(datum)

        # read the batch size
        import json
        params = json.load(file("%s/params.json"%(directory),"r"))
        batch_size = params["json_args"]["algo"]["batch_size"]


        if stat in list(progress.keys()):
            # trim to the shortest data curve
            n = np.amin([len(datum) for datum in data])
            data = [datum[:n] for datum in data]

            # plot the one-sigma region
            mean = np.mean(np.array(data),axis=0)
            xx = np.arange(len(mean)) * batch_size
            std = np.sqrt(np.var(np.array(data),axis=0))+ 1e-4 * mean # avoid zero stds
            ax.plot(xx,mean,color=color[:3])
            color2 = color
            color2[-1] = 0.3 # use a transparent color for filling
            ax.fill_between(xx,mean-std,mean+std,facecolor=color2,edgecolor="none")

    # pretty plot
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.legend(name_groups,bbox_to_anchor=(1.25,1), loc=9, ncol=1)
    plt.xlabel("total batch size")
    plt.title(stat)
    plt.show()
    return fig



# only reads float numbers...
def read_csv(csvfile):
    with open(csvfile) as f:
        reader = csv.DictReader(f)
        data = dict()
        for key in reader.fieldnames:
            data[key] = []
        for row in reader:
            for key in reader.fieldnames:
                value = row[key]
                data[key].append(float(value))
    return data

def load_problem(log_dir, iteration=None, pkl_file=None):
    if pkl_file is None:
        pkl_file_full =  "%s/itr_%d.pkl"%(log_dir,iteration)
    else:
        pkl_file_full = "%s/%s"%(log_dir,pkl_file)
    if os.path.isfile(pkl_file_full):
        data = joblib.load(pkl_file_full)
    else:
        print("Cannot find %s"%(pkl_file_full))
        sys.exit(1)

    # read algo (a bit cumbersome) ------------------------
    import json
    params = json.load(file("%s/params.json"%(log_dir),"r"))
    algo_spec = params["json_args"]["algo"]
    import imp
    _name = algo_spec['_name']
    script_name = _name.split('.')[2]
    class_name = _name.split('.')[3]
    algo_class = imp.load_source('rllab.algos.%s'%(script_name),'rllab/algos/%s.py'%(script_name))
    ALGO = getattr(algo_class,class_name)

    if "critic" in data:
        critic = data["critic"]
    else:
        critic = None

    if "resetter" in data:
        resetter = data["resetter"]
    else:
        from rllab.resetters.null_resetter import NullResetter
        resetter = NullResetter()

    algo = ALGO(env=data["env"],policy=data["policy"],baseline=data["baseline"],critic=critic,**algo_spec)
    data["algo"] = algo

    data["progress"] = read_csv("%s/progress.csv"%(log_dir))

    return data

def surf_pointmdp(mdp,variable, name,dx=0.1,interpolation="bilinear"):
    fig=plt.gcf()
    offset = dx*0.5
    cax = plt.imshow(np.flipud(variable),aspect="auto",extent=(-1-offset,1+offset,-1-offset,1+offset),interpolation=interpolation)
    mdp._mdp.draw_hole()
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.xlabel("x")
    plt.ylabel("y")
    fig.colorbar(cax,  orientation='vertical')
    plt.title(name)

# plot how the policy evolves over time
def plot_policy_evolution(log_dir,max_iter,stats,confidence=0.9,figsize=(7,7),duration=10):
    import matplotlib.pyplot as plt
    from IPython import display
    plt.figure(figsize=figsize)
    i=0
    while i < max_iter:
        display.clear_output(wait=True)
        t = time.clock()
        plt.clf()
        fig = plt.gcf()
        ax = fig.gca()

        data = joblib.load(("%s/itr_%d.pkl"%(log_dir,i)))
        mdp = data["mdp"]
        policy = data["policy"]

        if "means" in stats:
            filename = "%s/imgs/itr_%d_means.png"%(log_dir,i)
            ax.imshow(plt.imread(filename))
            plt.axis("off")
        if "vars" in stats:
            filename = "%s/imgs/itr_%d_vars.png"%(log_dir,i)
            ax.imshow(plt.imread(filename))
            plt.axis("off")
        if "policy" in stats:
            mdp._mdp.plot_policy(policy,confidence);

        #print "iteration: %d"%(i)

        display.display(plt.gcf())
        dt = time.clock()-t

        i = i + max_iter / (float(duration)/dt)
    display.clear_output(wait=True)


# rollout given a log directory
def rollout_given_log(dir_log,max_path_length):
    from rllab.sampler.utils import rollout

    data = joblib.load("%s/params.pkl"%(dir_log))
    policy = data['policy']
    mdp = data['mdp']

    rollout(mdp, policy, max_length=max_path_length, animated=True, speedup=5)
    mdp.stop_viewer()

# get a trajectory for a given policy and initial state
def rollout_given_state(mdp, policy, max_length,init_state=None,animated=False,speedup=5):
    observations = []
    states = []
    actions = []
    rewards = []
    pdists = []
    o = mdp._mdp.reset(init_state)
    policy.episode_reset()
    path_length = 0
    while path_length < max_length:
        states.append(mdp._mdp._full_state)
        observations.append(o)
        a, pdist = policy.get_action(o)
        actions.append(a)
        next_o, r, done = mdp.step(a)
        rewards.append(r)

        pdists.append(pdist)
        path_length += 1
        if done:
            break
        o = next_o
        if animated:
            mdp.plot()
            import time
            time.sleep(mdp.timestep / speedup)
    states.append(mdp._mdp._full_state)
    mdp.stop_viewer()
    return dict(
        observations=observations,
        states=states,
        actions=actions,
        rewards=rewards,
        pdists=pdists)


from rllab.misc import tensor_utils
def rollout(env, agent, max_path_length=np.inf, animated=False, speedup=1):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            env.render()
    return dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
    )

# given a series of states, animate the trajectory
def animate_traj(mdp,states,speedup=5):
    mdp.start_viewer()
    for i in range(len(states)):
        mdp._mdp.reset(states[i])
        mdp.plot()
        import time
        time.sleep(mdp.timestep / speedup)
    print(i)
    mdp.stop_viewer()

# generate trajectories and return
def gen_trajs(mdp,policy,init_state,max_path_length,n_rollout):
    import numpy as np
    results=[]
    for i in range(n_rollout):
        result = rollout_given_state(mdp, policy, max_length=max_path_length,init_state=init_state)
        returns.append(sum(rewards))
    return returns,np.var(returns)

# plot a statistics during the training progress; multiple trainings allowed

def plot_common(dirs,names,stat,stat2=None,xlim=None,ylim=None,figsize=None,smoothing=None):
    assert(len(dirs)==len(names))
    if figsize is not None:
        plt.figure(figsize=figsize)
    else:
        plt.figure()
    hsv = plt.get_cmap('hsv')
    colors = hsv(np.linspace(0, 1.0, len(names)+1))[:-1]
    ii = []
    for i in range(len(dirs)):
        progress = read_csv("%s/progress.csv"%(dirs[i]))
        # null resetters do not have test data
        if stat not in list(progress.keys()):
            stat3 = stat2
        else:
            stat3 = stat
        if stat3 in list(progress.keys()):
            ii.append(i)
            data = []
            for datum in progress[stat3]:
                if isinstance(datum,str): # sometimes a string may appear in progress.csv
                    try:
                        datum = float(datum)
                        data.append(datum)
                    except:
                        pass
                elif isinstance(datum,float) or isinstance(datum,int):
                    data.append(datum)
            data = np.array(data)
            if smoothing is not None:
                weights = np.repeat(1.0, smoothing)/smoothing
                data = np.convolve(data,weights,mode='same')
            plt.autoscale(tight=True)
            plt.plot(data,color=colors[i])
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.legend([names[i] for i in ii],bbox_to_anchor=(1.25,1), loc=9, ncol=1)
    plt.title(stat)
    plt.savefig('tmp.pdf',bbox_inches="tight")
    plt.show()



def plot_common_group(dir_groups,name_groups,stat,stat2=None,xlim=None,ylim=None,figsize=None,smoothing=None,new_progress_file="progress.csv"):
    assert(len(dir_groups)==len(name_groups))
    if figsize is not None:
        fig,ax = plt.subplots(figsize=figsize)
    else:
        fig,ax = plt.subplots()

    hsv = plt.get_cmap('hsv')
    colors = hsv(np.linspace(0, 1.0, len(name_groups)+1))[:-1]
    for i,color,dirs in zip(list(range(len(colors))),colors,dir_groups):
        data = []
        if len(dirs) == 0:
            print("Emptry directory list %s"%(name_groups[i]))
            sys.exit(1)
        for directory in dirs:
            import os
            file1 = "%s/progress.csv"%(directory)
            file2 = "%s/%s"%(directory,new_progress_file)
            if os.path.isfile(file2):
                progress_file = file2
            elif os.path.isfile(file1):
                progress_file = file1
            else:
                print("No progress.csv file for %s"%(directory))
                sys.exit(1)

            progress = read_csv(progress_file)
            if stat not in list(progress.keys()):
                stat3 = stat2
            else:
                stat3 = stat
            if stat3 not in list(progress.keys()):
                break
            datum = np.array(progress[stat3])
            if smoothing is not None:
                weights = np.repeat(1.0, smoothing)/smoothing
                datum = np.convolve(data,weights,mode='same')
            data.append(datum)
        if stat3 in list(progress.keys()):
            # trim to the shortest data curve
            n = np.amin([len(datum) for datum in data])
            data = [datum[:n] for datum in data]

            # plot the one-sigma region
            mean = np.mean(np.array(data),axis=0)
            std = np.sqrt(np.var(np.array(data),axis=0))+ 1e-4 * mean
            ax.plot(mean,color=color[:3])
            color2 = color
            color2[-1] = 0.3 # use a transparent color for filling
            ax.fill_between(np.arange(len(mean)),mean-std,mean+std,facecolor=color2,edgecolor="none")
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.legend(name_groups,bbox_to_anchor=(1.25,1), loc=9, ncol=1)
    plt.title(stat)
    plt.show()
    return fig


def plot_common_group_std(dir_groups,name_groups,stat,stat2=None,xlim=None,ylim=None,figsize=None,smoothing=None,new_progress_file="progress.csv"):
    assert(len(dir_groups)==len(name_groups))
    if figsize is not None:
        plt.subplots(figsize=figsize)
    else:
        plt.subplots()

    hsv = plt.get_cmap('hsv')
    colors = hsv(np.linspace(0, 1.0, len(name_groups)+1))[:-1]
    for color,dirs in zip(colors,dir_groups):
        data = []
        for directory in dirs:
            import os
            file1 = "%s/progress.csv"%(directory)
            file2 = "%s/%s"%(directory,new_progress_file)
            if os.path.isfile(file2):
                progress_file = file2
            else:
                progress_file = file1
            progress = read_csv(progress_file)
            if stat not in list(progress.keys()):
                stat3 = stat2
            else:
                stat3 = stat
            if stat3 not in list(progress.keys()):
                break
            datum = np.array(progress[stat3])
            if smoothing is not None:
                weights = np.repeat(1.0, smoothing)/smoothing
                datum = np.convolve(data,weights,mode='same')
            data.append(datum)
        if stat3 in list(progress.keys()):
            # trim to the shortest data curve
            n = np.amin([len(datum) for datum in data])
            data = [datum[:n] for datum in data]

            # plot the one-sigma region
            std = np.sqrt(np.var(np.array(data),axis=0))
            plt.plot(std,color=color[:3])
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.legend(name_groups,bbox_to_anchor=(1.25,1), loc=9, ncol=1)
    plt.title(stat + " (std) ")
    plt.show()


# plot the mean and std during the training progress; multiple trainings allowed
def plot_common_errorbar(dirs,names,mean,std,loc):
    assert(len(dirs)==len(names))
    plt.figure()
    xlim = float("inf")
    for i in range(len(dirs)):
        progress = read_csv("%s/progress.csv"%(dirs[i]))
        stat_mean = np.array(progress[mean])
        stat_std = np.array(progress[std])
        xlim = min(xlim,len(stat_mean))
        plt.errorbar(x=list(range(len(stat_mean))),
                 y=stat_mean,
                 yerr=stat_std)
    # plt.xlim([0,xlim])
    plt.legend(names,loc=loc)
    plt.title(mean)
    plt.show()

# generic object saving and loading
import pickle
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

from rllab.misc.special import discount_cumsum
def compute_return(path,gamma):
    path["returns"] = discount_cumsum(path["rewards"], gamma)
    return path["returns"]
