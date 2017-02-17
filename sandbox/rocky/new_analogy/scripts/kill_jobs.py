import json

import redis
import subprocess

from sandbox.rocky.cirrascale import cirra_config

redis_cli = redis.StrictRedis(host=cirra_config.REDIS_HOST)

from rllab.viskit.core import load_exps_data
from rllab import config
import os


exp_name = "fetch-relative-general-dagger-10"


def _kill_job(args):
    job_id, force = args
    try:
        ret = ""
        redis_cli = redis.StrictRedis(host=cirra_config.REDIS_HOST)
        content = redis_cli.get(job_id)
        job = json.loads(content.decode())
        print(job)

        assert len(job['job_id']) > 5

        command = [
            "ssh",
            "rocky@%s" % job['gpu_host'],
            "sudo kill -9 $(ps aux | grep %s | awk '{print $2}')" % job['job_id'],
        ]

        print(" ".join(command))

        ret = subprocess.check_output(command)
    except Exception as e:
        print(e)
        print(len(ret))
        if not force:
            return

    redis_cli.delete(job_id)
    print("Job %s deleted" % job_id)


exps = load_exps_data([os.path.join(config.PROJECT_PATH, "data/s3/bc-claw-2")])

jobs = redis_cli.keys("job/*")

job_list = []

for job in jobs:
    job_name = job.decode().split("/")[1]
    if "bc-claw-2" in job_name:
        exp = [exp for exp in exps if exp.params['exp_name'] == job_name][0]
        if exp.params['n_trajs'] in ['2k', '500']:
            job_list.append(job)

print(job_list)
import ipdb; ipdb.set_trace()

for job in job_list:
    _kill_job((job, False))
