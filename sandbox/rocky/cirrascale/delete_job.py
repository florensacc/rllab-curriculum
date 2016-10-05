from sandbox.rocky.cirrascale import client, cirra_config
import redis
import sys
import json
import os
import subprocess


job_id = sys.argv[1]
print(job_id)

redis_cli = redis.StrictRedis(host=cirra_config.REDIS_HOST)
job = json.loads(redis_cli.get("job/%s" % job_id).decode())
print(job)

assert len(job['job_id']) > 5

command = [
        "ssh",
        "rocky@%s" % job['gpu_host'],
        'sudo pkill -f -9 %s' % job['job_id']
]

print(command)

print(
    subprocess.check_output(command)
)

redis_cli.delete("job/%s" % job_id)
