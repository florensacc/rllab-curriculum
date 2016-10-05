from sandbox.rocky.cirrascale import client, cirra_config
import subprocess
import multiprocessing
import redis
import json


def get_gpu_type(host):
    try:
        result = subprocess.check_output([
            "ssh",
            "-oStrictHostKeyChecking=no",
            "-oConnectTimeout=10",
            "rocky@" + host, "nvidia-smi -L"
        ])
    except Exception as e:
        print("Error while probing %s" % host)
        return (host, None)
    result = result.decode()
    if "GeForce GTX TITAN X" in result:
        print("%s identified as TitanX Maxwell" % host)
        return (host, "maxwell")
    elif "TITAN X (Pascal)" in result:
        print("%s identified as TitanX Pascal" % host)
        return (host, "pascal")
    else:
        print(result)
        return (host, None)

if __name__ == "__main__":
    gpus = client.get_gpu_status()
    hosts = list(gpus.keys())
    directory = dict()
    redis_cli = redis.StrictRedis(host=cirra_config.REDIS_HOST)
    with multiprocessing.Pool() as pool:
        print("Probing status...")
        results = pool.map(get_gpu_type, hosts)
        results = dict([(k, v) for k, v in results if v is not None])
        print("Updating redis...")
        redis_cli.set(cirra_config.DIRECTORY_KEY, json.dumps(results))
        print("Updated")
