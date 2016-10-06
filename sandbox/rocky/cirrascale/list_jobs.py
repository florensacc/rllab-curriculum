from sandbox.rocky.cirrascale import client, cirra_config
import redis
import json


redis_cli = redis.StrictRedis(host=cirra_config.REDIS_HOST)
jobs = redis_cli.keys("job/*")

if len(jobs) == 0:
    print("No jobs currently running")
else:
    job_dicts = [json.loads(x.decode()) for x in redis_cli.mget(jobs)]
    job_dicts = sorted(job_dicts, key=lambda x: x["created_at"])

    for job in job_dicts:
        print("%s running on %s:%s" % (job["job_id"], job["gpu_host"], job["gpu_index"]))
