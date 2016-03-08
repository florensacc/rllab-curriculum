import re
import subprocess
import base64
import os.path as osp
import cPickle as pickle
from rllab.core.serializable import Serializable
from rllab import config
from rllab.misc.console import mkdir_p
from rllab.misc.ext import merge_dict
import datetime
import dateutil.tz
import json


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
            #mkstub = (lambda v_local: lambda *args, **kwargs: StubClass(v_local, *args, **kwargs))(v)
            #glbs[k].__new__ = types.MethodType(mkstub, glbs[k])
            #glbs[k].__init__ = types.MethodType(lambda *args: None, glbs[k])


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


def run_experiment_lite(stub_method_call, **kwargs):
    data = base64.b64encode(pickle.dumps(stub_method_call))
    script = kwargs.pop("script", "scripts/run_experiment_lite.py")
    mode = kwargs.pop("mode", "local")
    dry = kwargs.pop("dry", False)
    exp_prefix = kwargs.pop("exp_prefix")
    global exp_count
    exp_count += 1
    exp_name = "%s_%d" % (exp_prefix, exp_count)#, timestamp)
    kwargs["exp_name"] = exp_name
    kwargs["log_dir"] = config.LOG_DIR + "/" + exp_name
    if mode == "local":
        params = dict(kwargs.items() + [("args_data", data)])
        command = to_local_command(params, script=script)
        print(command)
        if dry:
            return
        try:
            subprocess.call(command, shell=True)
        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                raise
    elif mode == "local_docker":
        docker_image = kwargs.pop("docker_image", config.LOCAL_DOCKER_IMAGE)
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
        pass
    elif mode == "openai_kube":
        docker_image = kwargs.pop("docker_image", config.OPENAI_KUBE_DOCKER_IMAGE)
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
    else:
        raise NotImplementedError


_find_unsafe = re.compile(r'[a-zA-Z0-9_^@%+=:,./-]').search


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


def to_docker_command(params, docker_image, script='scripts/run_experiment.py', pre_commands=None, post_commands=None):
    """
    :param params: The parameters for the experiment. If logging directory parameters are provided, we will create
    docker volume mapping to make sure that the logging files are created at the correct locations
    :param docker_image: docker image to run the command on
    :param script: script command for running experiment
    :return:
    """
    log_dir = params.get("log_dir")
    mkdir_p(log_dir)
    # create volume for logging directory
    command_prefix = "docker run"
    docker_log_dir = config.DOCKER_LOG_DIR
    command_prefix += " -v {local_log_dir}:{docker_log_dir}".format(local_log_dir=log_dir, docker_log_dir=docker_log_dir)
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
    post_commands.append('aws s3 cp --recursive %s %s' % (config.DOCKER_LOG_DIR, osp.join(config.S3_PATH, params.get(
        "exp_name"))))
    command = to_docker_command(params, docker_image=docker_image, script=script, pre_commands=pre_commands,
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
