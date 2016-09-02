from rllab.misc.instrument import stub, run_experiment_lite
from rllab import config
import sys
import numpy as np
import os
import itertools
import git

from rllab.policies.categorical_tabular_q_policy import CategoricalTabularQPolicy
from rllab.envs.grid_world_env import GridWorldEnv
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.algos.tabular_q_learning import TabularQLearning

stub(globals())
os.path.join(config.PROJECT_PATH)

# check for uncommitted changes ------------------------------
repo = git.Repo('.')
if repo.is_dirty():
    answer = ''
    while answer not in ['y','Y','n','N']:
        answer = raw_input("The repository has uncommitted changes. Do you want to continue? (y/n)")
    if answer in ['n','N']:
        sys.exit(1)

# some important hyper-params that show up in the folder name -----------------
mode="local_test"
exp_prefix = os.path.basename(__file__).split('.')[0] # exp_xxx

# fixed params
init_eps = 1.0
init_q = 0.0
n_itr=100
batch_size=4000 # careful
max_path_length=400
discount=1.0
lr = 0.1
algo_type = "tabular_q"
eps_decay_scheme = "linear"
lr_decay_scheme = "linear"

# product params
desc_list = ['4x4']

# seeds
n_seed=5
seeds= np.arange(1,100*n_seed+1,100)

# mode-specific settings
if mode == "local_run":
    n_parallel = 4
    plot = True
elif mode == "local_test":
    n_parallel = 1
    plot = True
elif mode == "ec2":
    n_parallel = 1
    plot = False
elif mode == "ec2_parallel":
    n_parallel = 10
    config.AWS_INSTANCE_TYPE = "m4.10xlarge"
    config.AWS_SPOT_PRICE = '1.5'
    plot = False
else:
    raise NotImplementedError

# -------------------------------------------------------
exp_names = []
for desc in desc_list:
        env = GridWorldEnv(desc=desc)
        policy = CategoricalTabularQPolicy(
            env_spec=env.spec,
            init_eps=init_eps,
            init_q=init_q,
        )

        baseline = ZeroBaseline(env_spec=env.spec)

        if algo_type == "tabular_q":
            algo = TabularQLearning(
                env=env,
                policy=policy,
                baseline=baseline,
                n_itr=n_itr,
                batch_size=batch_size,
                max_path_length=max_path_length,
                discount=discount,
                gea_lambda=1,
                lr=lr,
                eps_decay_scheme = "linear",
                lr_decay_scheme = "linear",
            )
        else:
            raise NotImplementedError

        import datetime
        import dateutil.tz
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y%m%d_%H%M%S')

        exp_name_prefix="alex_{time}_{desc}_{algo_type}_eps{init_eps}_p{path_length}_lr{lr}_bs{batch_size}".format(
            time=timestamp,
            desc=desc,
            algo_type=algo_type,
            init_eps=init_eps,
            path_length=max_path_length,
            lr=lr,
            batch_size=batch_size,
        )
        exp_names.append("\"" + exp_name_prefix + "\",")

        for seed in seeds:
            exp_name = exp_name_prefix + "_s%d"%(seed)
            if "local" in mode:
                def run():
                    run_experiment_lite(
                        algo.train(),
                        exp_prefix=exp_prefix,
                        n_parallel=n_parallel,
                        snapshot_mode="all",
                        seed=seed,
                        plot=plot,
                        exp_name=exp_name,
                    )
            elif "ec2" in mode:
                if len(exp_name) > 64:
                    print "Should not use experiment name with length %d > 64.\nThe experiment name is %s.\n Exit now."%(len(exp_name),exp_name)
                    sys.exit(1)
                def run():
                    run_experiment_lite(
                        algo.train(),
                        exp_prefix=exp_prefix,
                        n_parallel=n_parallel,
                        snapshot_mode="last",
                        seed=seed,
                        plot=plot,
                        exp_name=exp_name,

                        mode="ec2",
                        terminate_machine=True,
                    )
            else:
                raise NotImplementedError
            run()
            if "test" in mode:
                sys.exit(0)


# record the experiment names to a file
# also record the branch name and commit number
logs = []
logs += ["branch: %s" %(repo.active_branch.name)]
logs += ["commit SHA: %s"%(repo.head.object.hexsha)]
logs += exp_names

cur_script_name = __file__
log_file_name = cur_script_name.split('.py')[0] + '.log'
with open(log_file_name,'w') as f:
    for message in logs:
        f.write(message + "\n")

# make the current script read-only to avoid accidental changes after ec2 runs
if "local" not in mode:
    os.system("chmod 444 %s"%(__file__))
