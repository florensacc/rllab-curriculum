from sandbox.rocky.cirrascale import client, cirra_config
import subprocess
import multiprocessing
import redis
import json


def get_gpu_type(host):
    print("Processing host %s" % host)
    try:
        result = subprocess.check_output([
            "ssh",
            "-oStrictHostKeyChecking=no",
            "-oConnectTimeout=10",
            "rocky@" + host,
            " && ".join([
                "sudo dpkg --configure -a",
                "sudo apt-get install -y awscli",
                "sudo sed -i '/rocky/s!\(.*:\).*:\(.*\)!\\1/local_home/rocky:\\2!' /etc/passwd",
                "sudo mkdir -p /local_home/rocky",
                "sudo chown -R rocky:rocky /local_home/rocky"
            ]),
        ])
    except Exception as e:
        print("Error while probing %s" % host)
        return None
        # return (host, None)
    # print(result)
    print("Successfully processed %s" % host)
    return None
    # result = result.decode()
    # if "GeForce GTX TITAN X" in result:
    #     print("%s identified as TitanX Maxwell" % host)
    #     return (host, "maxwell")
    # elif "TITAN X (Pascal)" in result:
    #     print("%s identified as TitanX Pascal" % host)
    #     return (host, "pascal")
    # else:
    #     print(result)
    #     return (host, None)

if __name__ == "__main__":
    gpus = client.get_gpu_status()
    hosts = sorted(list(gpus.keys()), key=lambda host: len([g for g in gpus[host] if g.available]))[::-1]
    directory = dict()
    redis_cli = redis.StrictRedis(host=cirra_config.REDIS_HOST)
    with multiprocessing.Pool(100) as pool:
        print("Probing status...")
        # host = "52.cirrascale.sci.openai.org"
        # print(hosts[0])
        # import ipdb; ipdb.set_trace()
        pool.map(get_gpu_type, hosts)##[0]])#hosts[0]])
        # results = dict([(k, v) for k, v in results if v is not None])
        # print("Updating redis...")
        # redis_cli.set(config.DIRECTORY_KEY, json.dumps(results))
        # print("Updated")