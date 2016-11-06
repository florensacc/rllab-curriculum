import multiprocessing
import os

import subprocess

from rllab import config
from sandbox.rocky.cirrascale.launch_job import get_directory


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


def update_exp_pkl(exp_prefix, exp_name):
    print("Searching for %s..." % exp_name)
    dir = get_directory()

    with multiprocessing.Pool(100) as pool:
        results = pool.starmap(check_host, [(host, exp_name) for host in dir.keys()])

    for result in results:
        if result is not None:
            host, paths = result
            paths = paths.split("\n")
            for path in paths:
                if len(path) > 0 and "data/local" in path:
                    local_path = os.path.join(
                        config.PROJECT_PATH,
                        "data/s3/{folder}/{exp_name}/params.pkl".format(folder=exp_prefix, exp_name=exp_name)
                    )
                    subprocess.check_call([
                        "ssh",
                        "rocky@{host}".format(host=host),
                        "cp {path}/params.pkl /tmp/{exp_name}.pkl".format(path=path, exp_name=exp_name)
                    ])
                    command = [
                        "scp",
                        "rocky@{host}:/tmp/{exp_name}.pkl".format(host=host, exp_name=exp_name),
                        local_path,
                    ]
                    print(" ".join(command))
                    subprocess.check_call(command)
                    return local_path
    raise FileNotFoundError("Not found: %s, %s" % (exp_prefix, exp_name))
