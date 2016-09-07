from rllab.misc.nb_utils import ExperimentDatabase
import matplotlib.pyplot as plt
import os
import joblib
import numpy as np
from glob import glob

def plot_reward(data_unpickle,color,fig_dir): 
    env = data_unpickle['env']
    ##retrieve original policy
    # poli = data_unpickle['policy']
    # mean = poli.get_action(np.array((0,)))[1]['mean']
    # logstd = poli.get_action(np.array((0,)))[1]['log_std'] 
    def normal(x): return 1/(np.exp(logstd)*np.sqrt(2*np.pi) )*np.exp(-0.5/np.exp(logstd)**2*(x-mean)**2) 
    x = np.arange(-2, 2, 0.01)
    reward = [env.reward_state(np.array([s])) for s in x]
    # policy_at0 = [normal(s) for s in x]
    plt.plot(x,reward,color=color, label= 'Reward function')
    # plt.plot(x,policy_at0,color=color*0.5,label='Policy at 0')
    plt.title('Reward acording to the state')
    plt.xlabel('state')
    plt.ylabel('Reward')
    plt.legend(loc='best')
    if fig_dir:
        plt.savefig(os.path.join(fig_dir,'Reward_function'))
    else:
        print("No directory for saving plots")

#Plot learning curve
def plot_learning_curve(exp,color,fig_dir):#######
    batch_size = exp.flat_params['json_args_algo_batch_size']
    lab = "bimod point mdp"
    plt.plot(exp.progress['AverageDiscountedReturn'], color=color,label = lab )
    plt.legend(loc='best')
    plt.title('Learning curve')
    plt.xlabel('iteration ({} steps each)'.format(batch_size))
    plt.ylabel('mean Reward')
    if fig_dir:
        plt.savefig(os.path.join(fig_dir,'learning_curve'))
    else:
        print("No directory for saving plots")


#final policy learned
def plot_policy_learned(data_unpickle, color, fig_dir=None):
    #recover the policy
    poli = data_unpickle['policy']
    #range to plot it
    x = np.arange(-3,3,0.01)
    means = np.zeros(np.size(x))
    logstd = np.zeros(np.size(x))
    for i,s in enumerate(x):
        means[i] = poli.get_action(np.array((s,)))[1]['mean']
        logstd[i] = poli.get_action(np.array((s,)))[1]['log_std']
        # means[i] = poli.get_action(np.array([s,]))[1]['mean']
        # logstd[i] = poli.get_action(np.array([s,]))[1]['log_std']

    plt.plot(x, means, color=color, label = 'mean')
    plt.plot(x, logstd, color=color * 0.7, label = 'logstd')
    plt.legend(loc = 5)
    plt.title('Final policy')
    plt.xlabel('state')
    plt.ylabel('Action')
    if fig_dir:
        plt.savefig(os.path.join(fig_dir,'policy_learned'))
    else:
        print("No directory for saving plots")

def plot_all_policy_at0(path_experiment,color,num_iter=100,fig_dir=None):
    mean_at_0 = []
    var_at_0 = []
    for itr in range(num_iter):
        data_bimodal_1d = joblib.load(os.path.join(path_experiment,'itr_{}.pkl'.format(itr)))
        poli = data_bimodal_1d['policy']
        action_at_0 = poli.get_action(np.array((0,)))
        mean_at_0.append(action_at_0[1]['mean'])
        var_at_0.append(action_at_0[1]['log_std'])
        # print "sampled action in iter {}: {}. Reward should be: {}".format(itr, action_at_0[0], reward(action_at_0[0]))
    itr = list(range(num_iter))
    plt.plot(itr,mean_at_0, color=color, label = 'mean at 0')
    plt.plot(itr, var_at_0, color=color * 0.7, label = 'logstd at 0')
    plt.title('How the policy variates accross iterations')
    plt.xlabel('iteration')
    plt.ylabel('mean and variance at 0')
    plt.legend(loc=3)
    if fig_dir:
        plt.savefig(os.path.join(fig_dir,'policy_progress'))
    else:
        print("No directory for saving plots")

## estimate by MC the policy at 0!
def plot_snn_at0(data_unpickle, itr=0, color=(1,0.1,0.1),fig_dir=None):
    #recover the policy
    poli = data_unpickle['policy']
    env  = data_unpickle['env']
    # x0 = np.zeros_like(env.wrapped_env.initial_state)
    #range to plot it
    bound = 3
    num_bins=600
    step = (2.*bound)/num_bins
    samples=num_bins*10
    x = np.arange(-bound,bound+step, step)
    px = np.zeros(np.size(x))
    for _ in range(samples):
        poli.reset()
        a = poli.get_action(np.array([0,]))[0]
        idx=int(np.floor(a/step) + bound/step)
        if idx>=0 and idx<len(px):
            px[idx]+=1
        elif idx<0:
            px[0]+=1
        else:
            px[len(px)-1]+=1

    px=px/float(samples)
    plt.plot(x, px, color=color, label = 'px')
    # plt.plot(x, logstd, color=color * 0.7, label = 'logstd')
    plt.legend(loc = 5)
    plt.title('Policy distribution at 0 after {} iter'.format(itr))
    plt.xlabel('next state')
    plt.ylabel('probability mass')
    if fig_dir:
        plt.savefig(os.path.join(fig_dir,'MC_policy_learned_at0_iter{}'.format(itr)))
    else:
        print("No directory for saving plots")


def plot_all_exp(datadir):
    database = ExperimentDatabase(datadir,names_or_patterns='*')
    exps = database._experiments
    colors=[(1,0.1,0.1),(0.1,1,0.1),(0.1,0.1,1),(1,1,0)]
    gather=1  # how many experiments to plot in the same image.

    for i, exp in enumerate(exps):
        #get the last pickle
        exp_name=exp.params['exp_name']
        path_experiment=os.path.join(datadir,exp_name)
        last_iter = np.size(exp.progress['Iteration']) - 1
        # pkl_name= 'itr_{}'.format(last_iter)
        # last_data_unpickle = joblib.load(os.path.join(path_experiment,pkl_name+'.pkl'))
        # first_data_unpickle = joblib.load(os.path.join(path_experiment,'itr_0.pkl'))
        pkl_file='params.pkl'
        last_data_unpickle  = joblib.load(os.path.join(path_experiment,pkl_file))
        first_data_unpickle = joblib.load(os.path.join(path_experiment,pkl_file))
        #create fig_dir
        fig_dir = os.path.join(path_experiment,'Figures')
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        #fix a color for plots of this exp
        color = np.array(colors[i%gather ])
        #plot everything
        print('Plotting for: ',exp_name)
        plt.figure( 1 + (i/gather)*6 )
        plot_reward(first_data_unpickle,color,fig_dir)
        print('Plotting learning curve')
        plt.figure(2 +(i/gather)*6 )
        plot_learning_curve(exp,color,fig_dir)
        print('Plotting last policy')
        plt.figure(3+(i/gather)*6 )
        plot_policy_learned(last_data_unpickle,color,fig_dir=fig_dir)
        # print 'Plotting policy progress'
        # plt.figure(4+(i/3)*6 )
        # plot_all_policy_at0(path_experiment,color,num_iter=last_iter+1,fig_dir=fig_dir)
        print('Plotting last policy MC')
        plt.figure(5+(i/gather)*6 )
        plot_snn_at0(last_data_unpickle, itr=last_iter+1, color=color, fig_dir=fig_dir)
        # print 'Plotting first policy MC'
        # plt.figure(6+(i/3)*6 )
        # plot_snn_at0(first_data_unpickle, itr=0, color=color,fig_dir=fig_dir)
        if (i+1)/gather > i/gather:
            plt.close('all')

## plot for all the experiments
if __name__ == "__main__":
    import sys
    name_dir=sys.argv[1]
    path_dir = "./data/local/"+name_dir
    print("plotting all experiments in: " +path_dir)
    plot_all_exp(path_dir)

