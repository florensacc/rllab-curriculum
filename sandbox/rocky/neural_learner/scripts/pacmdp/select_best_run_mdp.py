import multiprocessing
import os
import subprocess

from rllab import config
import os.path as osp
from rllab.viskit.core import load_exps_data
import numpy as np

from sandbox.rocky.cirrascale.launch_job import get_directory

exp_folders = [
    # osp.join(config.PROJECT_PATH, "data/s3/random-mdp-1"),
    # osp.join(config.PROJECT_PATH, "data/s3/random-mdp-10"),
    # osp.join(config.PROJECT_PATH, "data/s3/random-mdp-2"),
    # osp.join(config.PROJECT_PATH, "data/s3/random-mdp-3"),
    # osp.join(config.PROJECT_PATH, "data/s3/random-mdp-4"),
    # osp.join(config.PROJECT_PATH, "data/s3/random-mdp-5"),
    # osp.join(config.PROJECT_PATH, "data/s3/random-mdp-6"),
    # osp.join(config.PROJECT_PATH, "data/s3/random-mdp-7"),
    # osp.join(config.PROJECT_PATH, "data/s3/random-mdp-8"),
    # osp.join(config.PROJECT_PATH, "data/s3/random-mdp-9"),
    osp.join(config.PROJECT_PATH, "data/s3/random-mdp-12"),

]

for path in exp_folders:
    os.system("python scripts/sync_s3.py %s" % path.split("/")[-1])

exps = load_exps_data(exp_folders, ignore_missing_keys=True)

best_exp = dict()
best_exp_perf = dict()

for exp in exps:
    n_states = exp.params['n_states']
    n_actions = exp.params['n_actions']
    n_episodes = exp.params['n_episodes']
    episode_horizon = exp.params['episode_horizon']

    if n_states == 10 and n_actions == 5 and episode_horizon == 10:
        # if n_episodes in [10, 100, 500] and n_arms in [5, 10, 50]:
        if n_episodes in [10, 25, 50, 75, 100]:
            key = n_episodes  # (n_arms, n_episodes)
            if 'AverageReturn' not in exp.progress:
                continue
            perf = np.mean(exp.progress['AverageReturn'][-50:])
            if key not in best_exp_perf or best_exp_perf[key] < perf:
                if exp.params['exp_name'] not in [
                        'random-mdp-12_2016_10_28_17_32_40_0009', 'random-mdp-12_2016_10_27_01_21_11_0015']:
                    best_exp_perf[key] = perf
                    best_exp[key] = exp.params['exp_name']


def check_host(host, folder):
    try:
        command = [
            "ssh",
            "-oStrictHostKeyChecking=no",
            "-oConnectTimeout=10",
            "rocky@" + host,
            "find /local_home/rocky/rllab-workdir -name %s" % folder,
        ]
        # print(" ".join(command))
        file_location = subprocess.check_output(command).decode()
        if len(file_location) > 0:
            return (host, file_location)
    except Exception as e:
        pass
        # print(e)

    return None


for exp_name in best_exp.values():
    folder = exp_name.split("_")[0]

    local_path = os.path.join(config.PROJECT_PATH, "data/s3/{folder}/{exp_name}/params.pkl".format(folder=folder,
                                                                                                   exp_name=exp_name))

    if not os.path.exists(local_path):
        # search on cirrascale to see if file exists
        print("Searching for %s..." % exp_name)
        dir = get_directory()

        with multiprocessing.Pool(100) as pool:
            results = pool.starmap(check_host, [(host, exp_name) for host in dir.keys()])

        found = False

        for result in results:
            if result is not None:
                host, paths = result
                paths = paths.split("\n")
                for path in paths:
                    if len(path) > 0 and "data/local" in path:
                        command = [
                            "scp",
                            "rocky@{host}:{path}/params.pkl".format(host=host, path=path),
                            "data/s3/{folder}/{exp_name}/params.pkl".format(folder=folder, exp_name=exp_name)
                        ]
                        print(" ".join(command))
                        subprocess.check_call(command)
                        found = True
        if not found:
            print("%s still not found" % exp_name)
        else:
            print("%s found" % exp_name)
            # import ipdb;
            #
            # ipdb.set_trace()

print(best_exp)
print(best_exp_perf)


# run for even longer, see what happens?

# import ipdb; ipdb.set_trace()
