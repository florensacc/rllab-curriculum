from io import StringIO

from sandbox.rocky.cirrascale import client, cirra_config
from rllab import config
from rllab.misc import instrument
import redis
import json

DIRECTORY = None


def get_directory():
    global DIRECTORY
    if DIRECTORY is None:
        print("Loading directory from redis...")
        redis_cli = redis.StrictRedis(host=cirra_config.REDIS_HOST)
        DIRECTORY = json.loads(redis_cli.get(cirra_config.DIRECTORY_KEY).decode())
        print("Loaded")
    return DIRECTORY


CACHED_GPUS = None

FORBIDDEN = [5, 6, 7, 8, 9, 30, 54, 56]


def get_first_available_gpu(type_filter=None):
    global CACHED_GPUS
    if CACHED_GPUS is None:
        CACHED_GPUS = client.get_gpu_status()
    gpus = sorted(CACHED_GPUS.values(), key=lambda x: len([g for g in x if g.available]))[::-1]
    gpus = sum(gpus, [])
    gpus = [g for g in gpus if g.available]
    dir = get_directory()
    if type_filter is not None:
        gpus = [g for g in gpus if g.host in dir and dir[g.host] == type_filter]
    gpus = [g for g in gpus if int(g.host.split('.')[0]) not in FORBIDDEN]
    gpu = gpus[0]
    gpu.reserved = True
    gpu.force_available = False
    return gpu


def launch_cirrascale(type_filter="pascal"):
    def _launch(params, exp_prefix, docker_image=None, use_gpu=False, \
                python_command="python",
                script='scripts/run_experiment_lite.py', periodic_sync=True, sync_s3_pkl=False,
                sync_log_on_termination=True,
                periodic_sync_interval=15):
        assert use_gpu, "What a waste!"
        gpu = get_first_available_gpu(type_filter)

        if docker_image is None:
            docker_image = config.DOCKER_IMAGE
        code_full_path = instrument.s3_sync_code(config)
        code_file_name = code_full_path.split("/")[-1]

        job_id = params["exp_name"]

        local_code_path = "/local_home/rocky/rllab-workdir/%s" % job_id
        env = params.pop("env", None)
        if env is None:
            env = dict()

        params["log_dir"] = local_code_path + "/data/local/" + exp_prefix.replace("_", "-") + "/" + params[
            "exp_name"]
        log_dir = params.get("log_dir")
        remote_log_dir = params.pop("remote_log_dir")

        env = dict(env, CUDA_VISIBLE_DEVICES=str(gpu.index))

        sio = StringIO()
        sio.write("#!/bin/bash\n")
        sio.write("{\n")
        sio.write("""
            trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
        """.format(job_id=job_id))
        sio.write("""
            docker pull {docker_image}
        """.format(job_id=job_id, docker_image=docker_image))
        sio.write("""
            AWS_ACCESS_KEY_ID={aws_access_key}
            AWS_SECRET_ACCESS_KEY={aws_access_secret}
        """.format(job_id=job_id, aws_access_key=config.AWS_ACCESS_KEY, aws_access_secret=config.AWS_ACCESS_SECRET))
        sio.write("""
            aws s3 cp {code_full_path} /tmp/{code_file_name} --region {aws_region}
        """.format(job_id=job_id, code_full_path=code_full_path, local_code_path=local_code_path,
                   aws_region=config.AWS_REGION_NAME, code_file_name=code_file_name))
        sio.write("""
            mkdir -p {local_code_path}
        """.format(job_id=job_id, code_full_path=code_full_path, local_code_path=local_code_path,
                   aws_region=config.AWS_REGION_NAME))
        sio.write("""
            tar -zxvf /tmp/{code_file_name} -C {local_code_path}
        """.format(job_id=job_id, code_full_path=code_full_path, local_code_path=local_code_path,
                   aws_region=config.AWS_REGION_NAME, code_file_name=code_file_name))
        if periodic_sync:
            if sync_s3_pkl:
                sio.write("""
                    while /bin/true; do
                        aws s3 sync --exclude '*' --include '*.csv' --include '*.json' --include '*.pkl' {log_dir} {remote_log_dir} --region {aws_region}
                        sleep {periodic_sync_interval}
                    done & echo sync initiated""".format(log_dir=log_dir, remote_log_dir=remote_log_dir,
                                                         aws_region=config.AWS_REGION_NAME,
                                                         periodic_sync_interval=periodic_sync_interval))
            else:
                sio.write("""
                    while /bin/true; do
                        aws s3 sync --exclude '*' --include '*.csv' --include '*.json' {log_dir} {remote_log_dir} --region {aws_region}
                        sleep {periodic_sync_interval}
                    done & echo sync initiated""".format(log_dir=log_dir, remote_log_dir=remote_log_dir,
                                                         aws_region=config.AWS_REGION_NAME,
                                                         periodic_sync_interval=periodic_sync_interval))
        sio.write("""
            {command}
        """.format(
            command=instrument.to_docker_command(
                params,
                docker_image,
                use_gpu=use_gpu, env=env,
                local_code_dir=local_code_path,
                python_command=python_command,
                script=script,
                post_commands=[],
            ),
            job_id=job_id,
        ))
        sio.write("""
            aws s3 cp --recursive {log_dir} {remote_log_dir} --region {aws_region}
        """.format(log_dir=log_dir, remote_log_dir=remote_log_dir, aws_region=config.AWS_REGION_NAME))
        sio.write("""
            aws s3 cp {local_code_path}/user_data.log {remote_log_dir}/stdout.log --region {aws_region}
        """.format(local_code_path=local_code_path, remote_log_dir=remote_log_dir, aws_region=config.AWS_REGION_NAME))
        sio.write("""
            kill -9 $(jobs -p)
        """)
        sio.write("}} >> {local_code_path}/user_data.log 2>&1\n".format(local_code_path=local_code_path))

        # scp to /tmp and then execute directly
        full_script = instrument.dedent(sio.getvalue())
        s3_path = instrument.upload_file_to_s3(full_script)

        ssh_command = [
            "ssh",
            "-t",
            "-f",
            "rocky@" + gpu.host,
            "nohup sh -c \'" +
            " && ".join([
                            x.format(
                                local_code_path=local_code_path, s3_path=s3_path, aws_region=config.AWS_REGION_NAME,
                                aws_access_key=config.AWS_ACCESS_KEY,
                                aws_access_secret=config.AWS_ACCESS_SECRET
                            )
                            for x in [
                                "mkdir -p {local_code_path}",
                                "export AWS_ACCESS_KEY_ID=%s" % config.AWS_ACCESS_KEY,
                                "export AWS_SECRET_ACCESS_KEY=%s" % config.AWS_ACCESS_SECRET,
                                "aws s3 cp {s3_path} {local_code_path}/remote_script.sh --region {aws_region}",
                                "chmod +x {local_code_path}/remote_script.sh",
                                "bash {local_code_path}/remote_script.sh",
                            ]
                            ]) + " \' > /dev/null &",
        ]
        print(gpu)
        print(ssh_command)
        print("Job id: %s" % job_id)
        import subprocess

        # first submit the job to redis
        print("Submitting job to redis")
        redis_cli = redis.StrictRedis(host=cirra_config.REDIS_HOST)
        redis_cli.set("job/%s" % job_id, json.dumps(dict(
            job_id=job_id,
            gpu_host=gpu.host,
            gpu_index=gpu.index,
            created_at=instrument.timestamp,
        )))
        subprocess.check_call(ssh_command)

    return _launch
