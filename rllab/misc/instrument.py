import os
import re
import subprocess
import base64
import os.path as osp
import cPickle as pickle
from contextlib import contextmanager

import errno

from rllab.core.serializable import Serializable
from rllab import config
from rllab.misc.console import mkdir_p
from rllab.misc.ext import merge_dict
from StringIO import StringIO
import datetime
import dateutil.tz
import json

from rllab.viskit.core import flatten


class StubAttr(object):
    def __init__(self, obj, attr_name):
        self._obj = obj
        self._attr_name = attr_name

    @property
    def obj(self):
        return self._obj

    @property
    def attr_name(self):
        return self._attr_name

    def __call__(self, *args, **kwargs):
        return StubMethodCall(self.obj, self.attr_name, args, kwargs)


class StubMethodCall(Serializable):
    def __init__(self, obj, method_name, args, kwargs):
        Serializable.quick_init(self, locals())
        self.obj = obj
        self.method_name = method_name
        self.args = args
        self.kwargs = kwargs


class StubClass(object):
    def __init__(self, proxy_class):
        self.proxy_class = proxy_class

    def __call__(self, *args, **kwargs):
        if len(args) > 0:
            raise NotImplementedError
        return StubObject(self.proxy_class, *args, **kwargs)

    def __getstate__(self):
        return dict(proxy_class=self.proxy_class)

    def __setstate__(self, dict):
        self.proxy_class = dict["proxy_class"]

    def __getattr__(self, item):
        if hasattr(self.proxy_class, item):
            return StubAttr(self, item)
        raise AttributeError


class StubObject(object):
    def __init__(self, __proxy_class, *args, **kwargs):
        if len(args) > 0:
            raise NotImplementedError
        self.proxy_class = __proxy_class
        self.args = args
        self.kwargs = kwargs

    def __getstate__(self):
        return dict(args=self.args, kwargs=self.kwargs, proxy_class=self.proxy_class)

    def __setstate__(self, dict):
        self.args = dict["args"]
        self.kwargs = dict["kwargs"]
        self.proxy_class = dict["proxy_class"]

    def __getattr__(self, item):
        if hasattr(self.proxy_class, item):
            return StubAttr(self, item)
        raise AttributeError


def stub(glbs):
    # replace the __init__ method in all classes
    # hacky!!!
    for k, v in glbs.items():
        if isinstance(v, type) and v != StubClass:
            glbs[k] = StubClass(v)
            # mkstub = (lambda v_local: lambda *args, **kwargs: StubClass(v_local, *args, **kwargs))(v)
            # glbs[k].__new__ = types.MethodType(mkstub, glbs[k])
            # glbs[k].__init__ = types.MethodType(lambda *args: None, glbs[k])


# def run_experiment(params, script='scripts/run_experiment.py'):
#     command = to_command(params, script)
#     try:
#         subprocess.call(command, shell=True)
#     except Exception as e:
#         if isinstance(e, KeyboardInterrupt):
#             raise


exp_count = 0
now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')


def run_experiment_lite(
        stub_method_call,
        exp_prefix,
        script="scripts/run_experiment_lite.py",
        mode="local",
        dry=False,
        docker_image=None,
        aws_config=None,
        env=None,
        **kwargs):
    """
    Serialize the stubbed method call and run the experiment using the specified mode.
    :param stub_method_call: A stubbed method call.
    :param script: The name of the entrance point python script
    :param mode: Where & how to run the experiment. Should be one of "local", "local_docker", "ec2",
    "lab_kube" and "openai_kube" (must have OpenAI VPN set up).
    :param dry: Whether to do a dry-run, which only prints the commands without executing them.
    :param exp_prefix: Name prefix for the experiments
    :param docker_image: name of the docker image. Ignored if using local mode.
    :param aws_config: configuration for AWS. Only used under EC2 mode
    :param env: extra environment variables
    :param kwargs: All other parameters will be passed directly to the entrance python script.
    """
    data = base64.b64encode(pickle.dumps(stub_method_call))
    global exp_count
    exp_count += 1
    exp_name = "%s_%s_%04d" % (exp_prefix, timestamp, exp_count)
    kwargs["exp_name"] = exp_name
    kwargs["log_dir"] = config.LOG_DIR + "/local/" + exp_name
    kwargs["remote_log_dir"] = osp.join(config.AWS_S3_PATH, exp_prefix.replace("_", "-"),
                              exp_name)

    if mode == "local":
        # kwargs["log_dir"] = config.LOG_DIR + "/local/" + exp_name
        del kwargs["remote_log_dir"]
        params = dict(kwargs.items() + [("args_data", data)])
        command = to_local_command(params, script=script)
        print(command)
        if dry:
            return
        try:
            subprocess.call(command, shell=True, env=env)
        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                raise
    elif mode == "local_docker":
        if docker_image is None:
            docker_image = config.DOCKER_IMAGE
        params = dict(kwargs.items() + [("args_data", data)])
        command = to_docker_command(params, docker_image=docker_image, script=script)
        print(command)
        if dry:
            return
        try:
            subprocess.call(command, shell=True)
        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                raise
    elif mode == "ec2":
        if docker_image is None:
            docker_image = config.DOCKER_IMAGE
        params = dict(kwargs.items() + [("args_data", data)])
        launch_ec2(params, exp_prefix=exp_prefix, docker_image=docker_image, script=script,
                   aws_config=aws_config, dry=dry)
    elif mode == "openai_kube":
        if docker_image is None:
            docker_image = config.DOCKER_IMAGE
        params = dict(kwargs.items() + [("args_data", data)])
        pod_dict = to_openai_kube_pod(params, docker_image=docker_image, script=script)
        pod_str = json.dumps(pod_dict, indent=1)
        if dry:
            print(pod_str)
        fname = "{pod_dir}/{exp_prefix}/{exp_name}.json".format(
            pod_dir=config.POD_DIR,
            exp_prefix=exp_prefix,
            exp_name=exp_name
        )
        with open(fname, "w") as fh:
            fh.write(pod_str)
        kubecmd = "kubectl create -f %s" % fname
        print(kubecmd)
        if dry:
            return
        try:
            subprocess.call(kubecmd, shell=True)
        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                raise
    elif mode == "lab_kube":
        # first send code folder to s3
        s3_code_path = s3_sync_code(config, dry=dry)
        if docker_image is None:
            docker_image = config.DOCKER_IMAGE
        params = dict(kwargs.items() + [("args_data", data)])
        pod_dict = to_lab_kube_pod(params, code_full_path=s3_code_path, docker_image=docker_image, script=script)
        pod_str = json.dumps(pod_dict, indent=1)
        if dry:
            print(pod_str)
        dir = "{pod_dir}/{exp_prefix}".format(pod_dir=config.POD_DIR, exp_prefix=exp_prefix)
        ensure_dir(dir)
        fname = "{dir}/{exp_name}.json".format(
            dir=dir,
            exp_name=exp_name
        )
        with open(fname, "w") as fh:
            fh.write(pod_str)
        kubecmd = "kubectl create -f %s" % fname
        print(kubecmd)
        if dry:
            return
        try:
            subprocess.call(kubecmd, shell=True)
        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                raise
    else:
        raise NotImplementedError


_find_unsafe = re.compile(r'[a-zA-Z0-9_^@%+=:,./-]').search

def ensure_dir(dirname):
    """
    Ensure that a named directory exists; if it does not, attempt to create it.
    """
    try:
        os.makedirs(dirname)
    except OSError, e:
        if e.errno != errno.EEXIST:
            raise

def _shellquote(s):
    """Return a shell-escaped version of the string *s*."""
    if not s:
        return "''"

    if _find_unsafe(s) is None:
        return s

    # use single quotes, and put single quotes into double quotes
    # the string $'b is then quoted as '$'"'"'b'

    return "'" + s.replace("'", "'\"'\"'") + "'"


def _to_param_val(v):
    if v is None:
        return ""
    elif isinstance(v, list):
        return " ".join(map(_shellquote, map(str, v)))
    else:
        return _shellquote(str(v))


def to_local_command(params, script='scripts/run_experiment.py'):
    command = "python " + script
    for k, v in params.iteritems():
        if isinstance(v, dict):
            for nk, nv in v.iteritems():
                if str(nk) == "_name":
                    command += "  --%s %s" % (k, _to_param_val(nv))
                else:
                    command += \
                        "  --%s_%s %s" % (k, nk, _to_param_val(nv))
        else:
            command += "  --%s %s" % (k, _to_param_val(v))
    return command


def to_docker_command(params, docker_image, script='scripts/run_experiment.py', pre_commands=None,
                      post_commands=None, dry=False):
    """
    :param params: The parameters for the experiment. If logging directory parameters are provided, we will create
    docker volume mapping to make sure that the logging files are created at the correct locations
    :param docker_image: docker image to run the command on
    :param script: script command for running experiment
    :return:
    """
    log_dir = params.get("log_dir")
    if not dry:
        mkdir_p(log_dir)
    # create volume for logging directory
    command_prefix = "docker run"
    docker_log_dir = config.DOCKER_LOG_DIR
    command_prefix += " -v {local_log_dir}:{docker_log_dir}".format(local_log_dir=log_dir,
                                                                    docker_log_dir=docker_log_dir)
    params = merge_dict(params, dict(log_dir=docker_log_dir))
    command_prefix += " -t " + docker_image + " /bin/bash -c "
    command_list = list()
    if pre_commands is not None:
        command_list.extend(pre_commands)
    command_list.append("echo \"Running in docker\"")
    command_list.append(to_local_command(params, script))
    if post_commands is not None:
        command_list.extend(post_commands)
    return command_prefix + "'" + "; ".join(command_list) + "'"


def dedent(s):
    lines = [l.strip() for l in s.split('\n')]
    return '\n'.join(lines)


def launch_ec2(params, exp_prefix, docker_image, script='scripts/run_experiment.py',
               aws_config=None, dry=False):
    log_dir = params.get("log_dir")
    remote_log_dir = params.pop("remote_log_dir")


    default_config = dict(
        image_id=config.AWS_IMAGE_ID,
        instance_type=config.AWS_INSTANCE_TYPE,
        key_name=config.AWS_KEY_NAME,
        spot=config.AWS_SPOT,
        spot_price=config.AWS_SPOT_PRICE,
        iam_instance_profile_name=config.AWS_IAM_INSTANCE_PROFILE_NAME,
        security_groups=config.AWS_SECURITY_GROUPS,
    )

    if aws_config is None:
        aws_config = dict()
    aws_config = merge_dict(default_config, aws_config)

    sio = StringIO()
    sio.write("#!/bin/bash\n")
    sio.write("rm /home/ubuntu/user_data.log")
    sio.write("{\n")
    sio.write("""
        die() { status=$1; shift; echo "FATAL: $*"; exit $status; }
    """)
    sio.write("""
        EC2_INSTANCE_ID="`wget -q -O - http://instance-data/latest/meta-data/instance-id`"
    """)
    sio.write("""
        aws ec2 create-tags --resources $EC2_INSTANCE_ID --tags Key=Name,Value={exp_name} --region {aws_region}
    """.format(exp_name=params.get("exp_name"), aws_region=config.AWS_REGION_NAME))
    sio.write("""
        service docker start
    """)
    sio.write("""
        DOCKER_HOST=\"tcp://localhost:4243\" docker --config /home/ubuntu/.docker pull {docker_image}
    """.format(docker_image=docker_image))
    sio.write("""
        mkdir -p {log_dir}
    """.format(log_dir=log_dir))
    sio.write("""
        while /bin/true; do
            aws s3 sync --exclude *.pkl --exclude *.log {log_dir} {remote_log_dir} --region {aws_region}
            sleep 1
        done &
    """.format(log_dir=log_dir, remote_log_dir=remote_log_dir, aws_region=config.AWS_REGION_NAME))
    sio.write("""
        DOCKER_HOST=\"tcp://localhost:4243\" {command}
    """.format(command=to_docker_command(params, docker_image, script)))
    sio.write("""
        aws s3 cp --recursive {log_dir} {remote_log_dir} --region {aws_region}
    """.format(log_dir=log_dir, remote_log_dir=remote_log_dir, aws_region=config.AWS_REGION_NAME))
    sio.write("""
        aws s3 cp /home/ubuntu/user_data.log {remote_log_dir}/stdout.log --region {aws_region}
    """.format(remote_log_dir=remote_log_dir, aws_region=config.AWS_REGION_NAME))
    sio.write("""
        EC2_INSTANCE_ID="`wget -q -O - http://instance-data/latest/meta-data/instance-id || die \"wget instance-id has failed: $?\"`"
        aws ec2 terminate-instances --instance-ids $EC2_INSTANCE_ID --region {aws_region}
    """.format(aws_region=config.AWS_REGION_NAME))
    sio.write("} >> /home/ubuntu/user_data.log 2>&1\n")

    import boto3
    import botocore
    if aws_config["spot"]:
        ec2 = boto3.client(
            "ec2",
            region_name=config.AWS_REGION_NAME,
            aws_access_key_id=config.AWS_ACCESS_KEY,
            aws_secret_access_key=config.AWS_ACCESS_SECRET,
        )
    else:
        ec2 = boto3.resource(
            "ec2",
            region_name=config.AWS_REGION_NAME,
            aws_access_key_id=config.AWS_ACCESS_KEY,
            aws_secret_access_key=config.AWS_ACCESS_SECRET,
        )
    instance_args = dict(
        ImageId=aws_config["image_id"],
        KeyName=aws_config["key_name"],
        UserData=dedent(sio.getvalue()),
        InstanceType=aws_config["instance_type"],
        EbsOptimized=True,
        SecurityGroups=aws_config["security_groups"],
        IamInstanceProfile=dict(
            Name=aws_config["iam_instance_profile_name"],
        ),
    )
    if not aws_config["spot"]:
        instance_args["MinCount"] = 1
        instance_args["MaxCount"] = 1
    print "************************************************************"
    print instance_args["UserData"]
    print "************************************************************"
    if aws_config["spot"]:
        instance_args["UserData"] = base64.b64encode(instance_args["UserData"])
        spot_args = dict(
            DryRun=dry,
            InstanceCount=1,
            LaunchSpecification=instance_args,
            SpotPrice=aws_config["spot_price"],
            ClientToken=params["exp_name"],
        )
        import pprint
        pprint.pprint(spot_args)
        if not dry:
            response = ec2.request_spot_instances(**spot_args)
            print response
            spot_request_id = response['SpotInstanceRequests'][0]['SpotInstanceRequestId']
            for _ in range(10):
                try:
                    ec2.create_tags(
                        Resources=[spot_request_id],
                        Tags=[{'Key': 'Name', 'Value': params["exp_name"]}],
                    )
                    break
                except botocore.exceptions.ClientError:
                    continue
    else:
        import pprint
        pprint.pprint(instance_args)
        ec2.create_instances(
            DryRun=dry,
            **instance_args
        )


def to_openai_kube_pod(params, docker_image, script='scripts/run_experiment.py'):
    """
    :param params: The parameters for the experiment. If logging directory parameters are provided, we will create
    docker volume mapping to make sure that the logging files are created at the correct locations
    :param docker_image: docker image to run the command on
    :param script: script command for running experiment
    :return:
    """
    log_dir = params.get("log_dir")
    mkdir_p(log_dir)
    pre_commands = list()
    pre_commands.append('mkdir -p ~/.aws')
    # fetch credentials from the kubernetes secret file
    pre_commands.append('echo "[default]" >> ~/.aws/credentials')
    pre_commands.append(
        'echo "aws_access_key_id = $(cat /etc/rocky-s3-credentials/access-key)" >> ~/.aws/credentials')
    pre_commands.append(
        'echo "aws_secret_access_key = $(cat /etc/rocky-s3-credentials/access-secret)" >> ~/.aws/credentials')
    # copy the file to s3 after execution
    post_commands = list()
    post_commands.append('aws s3 cp --recursive %s %s' %
                         (config.DOCKER_LOG_DIR,
                          osp.join(config.AWS_S3_PATH, params.get("exp_name"))))
    command = to_docker_command(params, docker_image=docker_image, script=script,
                                pre_commands=pre_commands,
                                post_commands=post_commands)
    pod_name = config.KUBE_PREFIX + params["exp_name"]
    # underscore is not allowed in pod names
    pod_name = pod_name.replace("_", "-")
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": pod_name,
            "labels": {
                "expt": pod_name,
            },
        },
        "spec": {
            "containers": [
                {
                    "name": "foo",
                    "image": docker_image,
                    "command": ["/bin/bash", "-c", command],
                    "resources": {
                        "limits": {
                            "cpu": "4"
                        },
                    },
                    "imagePullPolicy": "Always",
                    "volumeMounts": [{
                        "name": "rocky-s3-credentials",
                        "mountPath": "/etc/rocky-s3-credentials",
                        "readOnly": True
                    }],
                }
            ],
            "volumes": [{
                "name": "rocky-s3-credentials",
                "secret": {
                    "secretName": "rocky-s3-credentials"
                },
            }],
            "imagePullSecrets": [{"name": "quay-login-secret"}],
            "restartPolicy": "Never",
            "nodeSelector": {
                "aws/class": "m",
                "aws/type": "m4.xlarge",
            }
        }
    }

S3_CODE_PATH = None
def s3_sync_code(config, dry=False):
    global S3_CODE_PATH
    if S3_CODE_PATH is not None:
        return S3_CODE_PATH
    base = config.AWS_CODE_SYNC_S3_PATH
    has_git = True
    try:
        current_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip()
        clean_state = len(
            subprocess.check_output(["git", "status", "--porcelain"])) == 0
    except subprocess.CalledProcessError as _:
        print "Warning: failed to execute git commands"
        has_git = False
    dir_hash = base64.b64encode(subprocess.check_output(["pwd"]))
    code_path = "%s_%s" % (
        dir_hash,
        (current_commit if clean_state else "%s_dirty_%s" % (current_commit, timestamp)) if
            has_git else timestamp
    )
    full_path = "%s/%s" % (base, code_path)
    cache_path = "%s/%s" % (base, dir_hash)
    cache_cmds = ["aws", "s3", "sync"] + \
                 [cache_path, full_path]
    cmds = ["aws", "s3", "sync"] + \
           flatten(["--exclude", "%s" % pattern] for pattern in config.CODE_SYNC_IGNORES) + \
           [".", full_path]
    caching_cmds = ["aws", "s3", "sync"] + \
                   [full_path, cache_path]
    print cache_cmds, cmds, caching_cmds
    if not dry:
        subprocess.check_call(cache_cmds)
        subprocess.check_call(cmds)
        subprocess.check_call(caching_cmds)
    S3_CODE_PATH = full_path
    return full_path

def to_lab_kube_pod(params, docker_image, code_full_path, script='scripts/run_experiment.py'):
    """
    :param params: The parameters for the experiment. If logging directory parameters are provided, we will create
    docker volume mapping to make sure that the logging files are created at the correct locations
    :param docker_image: docker image to run the command on
    :param script: script command for running experiment
    :return:
    """
    log_dir = params.get("log_dir")
    remote_log_dir = params.pop("remote_log_dir")
    mkdir_p(log_dir)
    pre_commands = list()
    pre_commands.append('mkdir -p ~/.aws')
    # fetch credentials from the kubernetes secret file
    pre_commands.append('echo "[default]" >> ~/.aws/credentials')
    pre_commands.append(
        "echo \"aws_access_key_id = %s\" >> ~/.aws/credentials" % config.AWS_ACCESS_KEY)
    pre_commands.append(
        "echo \"aws_secret_access_key = %s\" >> ~/.aws/credentials" % config.AWS_ACCESS_SECRET)
    pre_commands.append('aws s3 cp --recursive %s %s' %
                         (code_full_path, config.DOCKER_CODE_DIR))
    pre_commands.append('cd %s' %
                        (config.DOCKER_CODE_DIR))
    pre_commands.append("""
        while /bin/true; do
            aws s3 sync --exclude *.pkl --exclude *.log {log_dir} {remote_log_dir} --region {aws_region}
            sleep 5
        done & echo sync initiated""".format(log_dir=log_dir, remote_log_dir=remote_log_dir, aws_region=config.AWS_REGION_NAME))
    # copy the file to s3 after execution
    post_commands = list()
    post_commands.append('aws s3 cp --recursive %s %s' %
                         (log_dir,
                          remote_log_dir))
    # command = to_docker_command(params, docker_image=docker_image, script=script,
    #                             pre_commands=pre_commands,
    #                             post_commands=post_commands)
    command_list = list()
    if pre_commands is not None:
        command_list.extend(pre_commands)
    command_list.append("echo \"Running in docker\"")
    command_list.append(to_local_command(params, script))
    if post_commands is not None:
        command_list.extend(post_commands)
    command = "; ".join(command_list)
    pod_name = config.KUBE_PREFIX + params["exp_name"]
    # underscore is not allowed in pod names
    pod_name = pod_name.replace("_", "-")
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": pod_name,
            "labels": {
                "expt": pod_name,
            },
        },
        "spec": {
            "containers": [
                {
                    "name": "foo",
                    "image": docker_image,
                    "command": ["/bin/bash", "-c", command],
                    "resources": {
                        "requests": {
                            "cpu": 2,
                        }
                    },
                    "imagePullPolicy": "Always",
                }
            ],
            "restartPolicy": "Never",
            "nodeSelector": {
                "aws/type": "m4.xlarge",
            }
        }
    }
