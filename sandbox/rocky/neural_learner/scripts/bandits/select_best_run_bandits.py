import multiprocessing
import os
import subprocess

from rllab import config
import os.path as osp
from rllab.viskit.core import load_exps_data
import numpy as np

from sandbox.rocky.cirrascale.launch_job import get_directory

exp_folders = [
    # osp.join(config.PROJECT_PATH, "data/s3/mab-1"),
    # osp.join(config.PROJECT_PATH, "data/s3/mab-2"),
    # osp.join(config.PROJECT_PATH, "data/s3/mab-3"),
    # osp.join(config.PROJECT_PATH, "data/s3/mab-4"),
    # osp.join(config.PROJECT_PATH, "data/s3/mab-5"),
    # osp.join(config.PROJECT_PATH, "data/s3/mab-5-1"),
    # osp.join(config.PROJECT_PATH, "data/s3/mab-6"),
    # osp.join(config.PROJECT_PATH, "data/s3/mab-7"),
    # osp.join(config.PROJECT_PATH, "data/s3/mab-8"),
    # osp.join(config.PROJECT_PATH, "data/s3/mab-9"),
    # # osp.join(config.PROJECT_PATH, "data/s3/mab-9-1"),
    # # osp.join(config.PROJECT_PATH, "data/s3/mab-9-2"),
    # osp.join(config.PROJECT_PATH, "data/s3/mab-10"),
    # osp.join(config.PROJECT_PATH, "data/s3/mab-10-1"),
    # osp.join(config.PROJECT_PATH, "data/s3/mab-11"),
    # osp.join(config.PROJECT_PATH, "data/s3/mab-11-1"),
    # osp.join(config.PROJECT_PATH, "data/s3/mab-11-2"),
    osp.join(config.PROJECT_PATH, "data/s3/mab-17"),
]

exps = load_exps_data(exp_folders)

best_exp = dict()
best_exp_perf = dict()

for exp in exps:
    n_episodes = exp.params['n_episodes']
    n_arms = exp.params['n_arms']
    if n_episodes in [10, 100, 500] and n_arms in [5, 10, 50]:
        key = (n_arms, n_episodes)
        if 'AverageReturn' not in exp.progress:
            continue
        perf = np.mean(exp.progress['AverageReturn'][-50:])
        if key not in best_exp_perf or best_exp_perf[key] < perf:
            best_exp_perf[key] = perf
            best_exp[key] = exp.params['exp_name']


def check_host(host, folder):
    try:
        command = [
            "ssh",
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
    dir = get_directory()

    if not os.path.exists(local_path):
        # search on cirrascale to see if file exists
        print("Searching for %s..." % exp_name)

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
