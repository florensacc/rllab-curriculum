#!/usr/bin/env python
from edison import (
    ScriptRun,assert_script_runs_different,ordered_load,
    load_config, dict_update, filter_dict, string2dict,
    maybe_call_and_print, yes_or_no, check_output_and_print
    )
import os.path as osp
from fnmatch import fnmatch
from collections import OrderedDict
import argparse, json, pipes, shutil, time
import random, numpy as np

def filter_dict_by_desc(d,desc):
    if "," in desc:
        scriptnames = desc.split(",")
        filtfn = lambda s: s in scriptnames
    elif "*" in desc:
        filtfn = lambda s: fnmatch(s, desc)
    else:
        filtfn = lambda s: s==desc
    return filter_dict(d, lambda key,_: filtfn(key))

def oneof(*args):
    s = 0
    for arg in args: s += bool(arg)
    return s==1

def get_vmdir():
    return "/tmp/expt"

def make_full_cmd(expt, config, sr, pipetype, upload=False):    
    cmds = ["mkdir -p %s"%get_vmdir(), sr.get_cmd(pipe_to_logfile=pipetype)] 
    if upload:
        cmds.append('aws s3 cp --recursive %s %s'%(get_vmdir(), osp.join(config['aws']["s3_path"],expt)))
    cmd = " ; ".join(cmds)
    return cmd

def dedent(s):
    lines = [l.strip() for l in s.split('\n')]
    return '\n'.join(lines)

def make_pod(config, expt_name, index, command):
    podparts = []
    prefix = config["instance_prefix"]
    if prefix: podparts.append(prefix)
    podparts.extend([expt_name, "%.4i"%index])
    podname="-".join(podparts)

    command = ["/bin/bash", "-c", command]
    return {
        "apiVersion" : "v1",
        "kind" : "Pod",
        "metadata" : {
            "name" : podname,
            "labels": {
              "expt": expt_name
            },
        },
        "spec" : {
            "containers" : [
                {
                    "name" : "foo",
                    "image" : config["container"],
                    "command" : command,
                    "resources" : {"requests" : {"cpu" : "1"}},
                    "imagePullPolicy" : "Always",
                }
            ],
            "imagePullSecrets" : [{"name":"quay-login-secret"}],
            "restartPolicy" : "Never",
            "nodeSelector" : {"aws/class": "m"}
        }
    }




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_file",type=argparse.FileType("r"))
    parser.add_argument("expt_name")
    parser.add_argument("--script_include")
    parser.add_argument("--cfg_include")
    parser.add_argument("--test_include")

    parser.add_argument("--mode",choices=["local-docker", "local", "cloud"])
    parser.add_argument("--upload",type=int,default=1)
    parser.add_argument("--n_runs",type=int)
    parser.add_argument("--kwargs",type=str)
    parser.add_argument("--cfg_name_prefix",type=str,default="")
    parser.add_argument("--pipe_to_logfile",choices=["off","all","stdout"],default="all")
    parser.add_argument("--start_from")
    parser.add_argument("--start_run",type=int,default=0)
    parser.add_argument("--start_idx",type=int,default=0)

    parser.add_argument("--test",action="store_true")


    config = load_config()
    parser.add_argument("--dry",action="store_true")
    args = parser.parse_args()

    assert args.mode is not None

    expt_info = ordered_load(args.yaml_file)
    n_runs = args.n_runs or expt_info.get("n_runs",1)
    default_settings = expt_info.get("default_settings",{})
    if args.kwargs: default_settings.update(string2dict(args.kwargs))
    if args.test: 
        args.pipe_to_logfile = "off"
        n_runs = 1

    if 'cfg_name' in default_settings: default_settings['cfg_name'] = str(default_settings['cfg_name']) # Deal with stupid bug where commit name gets turned into int

    assert "scripts" in expt_info or ("tests" in expt_info and ("cfgs" in expt_info or "sampling" in expt_info))

    if "sampling" in expt_info:        
        sampling = expt_info["sampling"]
        n_samples = sampling.pop("n_samples")
        cfg_dict = expt_info["cfgs"] = OrderedDict()
        for i in xrange(n_samples):
            cfgname = "random%.4i"%i
            d = cfg_dict[cfgname] = {}
            for (pname, sinfo) in sampling.items():
                stype = sinfo[0]
                sparams = sinfo[1:]
                if stype == "C":
                    d[pname] = random.choice(sparams)
                elif stype == "U":
                    d[pname] = random.uniform(*sparams)
                elif stype == "Ulog":
                    low,high = sparams
                    d[pname] = np.exp(random.uniform(np.log(low), np.log(high)))

    if "scripts" in expt_info:
        script_dict = expt_info["scripts"]
        if args.script_include is not None:
            script_dict = filter_dict_by_desc(script_dict,args.script_include)
        assert args.test_include is None and args.cfg_include is None,"can't use {cfg/test}_include when yaml file has scripts:"
    else:
        test_dict = expt_info["tests"]
        if args.test_include: test_dict = filter_dict_by_desc(test_dict,args.test_include)
        cfg_dict = expt_info["cfgs"]
        if args.cfg_include: cfg_dict = filter_dict_by_desc(cfg_dict,args.cfg_include)
        script_dict = {}
        for (testname,testinfo) in test_dict.items():
            for (cfgname,cfginfo) in cfg_dict.items():
                scriptinfo = {}
                scriptinfo["test_name"] = testname
                scriptinfo["cfg_name"] = args.cfg_name_prefix + cfgname
                scriptinfo.update(testinfo)
                scriptinfo.update(cfginfo)
                scriptname = cfgname + "-" + testname
                script_dict[scriptname] = scriptinfo
        assert args.script_include is None,"For this type of yaml file, you should use cfg_include/test_include"


    if args.start_from is not None:
        pairs = []
        gotit = False
        for (k,v) in script_dict.items():
            if k==args.start_from:
                gotit=True
            if gotit:
                pairs.append((k,v))
            else:
                print "skipping",k
        script_dict = OrderedDict(pairs)
    assert len(script_dict) > 0


    vmdir = get_vmdir()
    if args.mode=="local":
        if osp.exists(vmdir) and yes_or_no("%s exists. delete?" % vmdir):
            if not args.dry: shutil.rmtree(vmdir)
        md = {}
    else:
        dockercommit = check_output_and_print("docker images -q %s"%config["container"]).strip()
        md = {"dockercommit" : dockercommit}

    all_srs = []
    for i_run in xrange(args.start_run, n_runs):
        for (script_name,script_info) in script_dict.items():
            script_info = dict_update(default_settings, script_info)
            sr = ScriptRun(script_info,script_name,i_run, vmdir, md)
            all_srs.append(sr)

    assert_script_runs_different(all_srs)
    # make_bash_script(expt_name, config, srs, local=False, shutdown_instance=False, upload_result=False):
    if args.mode=="cloud":
        for (index,sr) in enumerate(all_srs): 
            import time; time.sleep(.1)
            cmd = make_full_cmd(args.expt_name, config, sr, args.pipe_to_logfile, upload=True)        
            pod_desc = make_pod(config, args.expt_name, index+args.start_idx, cmd)
            print pod_desc
            fname = "/tmp/pod.json"
            podstr = json.dumps(pod_desc, indent=1)
            with open(fname,"w") as fh: fh.write(podstr)        
            kubecmd = "kubectl create -f %s"%fname
            maybe_call_and_print(kubecmd, args.dry)
        print "started %i jobs"%len(all_srs)
    elif args.mode=="local-docker":
        for sr in all_srs:
            print "**************************"
            cmd = make_full_cmd(args.expt_name, config, sr, args.pipe_to_logfile, upload=args.upload)
            dockercmd = "docker run -t %s /bin/bash -c %s"%(config["container"], pipes.quote(cmd))
            maybe_call_and_print(dockercmd, args.dry)
    elif args.mode=="local":
        for sr in all_srs:
            print "**************************"
            cmd = make_full_cmd(args.expt_name, config, sr, args.pipe_to_logfile, upload=False)
            maybe_call_and_print(cmd, args.dry)





if __name__ == "__main__":
    main()