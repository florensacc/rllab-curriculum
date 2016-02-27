#!/usr/bin/env python
from edison import (
    ScriptRun,assert_script_runs_different,ordered_load,
    load_config,
    chunkify, dict_update, filter_dict,
    get_git_commit,
    maybe_call_and_print,
    check_output_and_print,
    random_string, string2dict)

import subprocess
import os.path as osp
from fnmatch import fnmatch
from collections import OrderedDict
import argparse
from StringIO import StringIO
import pipes
import tempfile

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

def make_bash_script(expt_name, config, srs, cloud=True, shutdown_instance=False, upload=False):

    if cloud:
        hostdir = osp.join("/tmp/experiments",expt_name)
    else:
        hostdir = osp.join(config["expt_dir"],expt_name)

    vmdir = get_vmdir()

    container = config["container"]
    sio = StringIO()
    sio.write("#!/bin/bash\n")
    # sio.write("set -e\n")
    if cloud:
        sio.write('docker pull %s\n'%container)
    sio.write("mkdir -p %s\n"%hostdir)
    for sr in srs:
        cmd = sr.get_cmd(pipe_to_logfile="all")
        sio.write("docker run -t -v %s:%s:rw %s /bin/bash -c %s\n"%(hostdir,vmdir,container, pipes.quote(cmd)))


    if upload:
        sio.write('aws s3 cp --recursive %s %s\n'%(hostdir, osp.join(config['aws']["s3_path"],expt_name)))
    if shutdown_instance:
        assert cloud
        sio.write("""
            die() { status=$1; shift; echo "FATAL: $*"; exit $status; }
            EC2_INSTANCE_ID="`wget -q -O - http://instance-data/latest/meta-data/instance-id || die \"wget instance-id has failed: $?\"`"
            aws ec2 terminate-instances --instance-ids $EC2_INSTANCE_ID
        """)


    return dedent(sio.getvalue())

def dedent(s):
    lines = [l.strip() for l in s.split('\n')]
    return '\n'.join(lines)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_file",type=argparse.FileType("r"))
    parser.add_argument("expt_name")
    parser.add_argument("--script_include")
    parser.add_argument("--cfg_include")
    parser.add_argument("--test_include")

    parser.add_argument("--cloud",action="store_true")
    parser.add_argument("--one_script_per_machine",action="store_true")
    parser.add_argument("--one_run_per_machine",action="store_true")
    parser.add_argument("--scripts_per_machine",type=int)
    parser.add_argument("--keep_instance",action="store_true")
    parser.add_argument("--instance_prefix")
    parser.add_argument("--n_runs",type=int)
    parser.add_argument("--alter_default_settings",type=str)
    parser.add_argument("--cfg_name_prefix",type=str,default="")
    parser.add_argument("--pipe_to_logfile",choices=["off","all","stdout"],default="all")
    parser.add_argument("--start_from")
    parser.add_argument("--start_run",type=int,default=0)

    parser.add_argument("--test",action="store_true")


    config = load_config()
    parser.add_argument("--dry",action="store_true")
    args = parser.parse_args()

    if not args.test: assert oneof(args.one_script_per_machine,args.one_run_per_machine,args.scripts_per_machine)
    
    expt_info = ordered_load(args.yaml_file)
    n_runs = args.n_runs or expt_info.get("n_runs",1)
    default_settings = expt_info.get("default_settings",{})
    if args.alter_default_settings: default_settings.update(string2dict(args.alter_default_settings))
    if args.test: 
        args.pipe_to_logfile = "off"
        n_runs = 1

    if 'cfg_name' in default_settings: default_settings['cfg_name'] = str(default_settings['cfg_name']) # Deal with stupid bug where commit name gets turned into int

    assert "scripts" in expt_info or ("tests" in expt_info and "cfgs" in expt_info)

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
    if args.cloud:
        dockercommit = check_output_and_print("docker images -q %s"%config["container"]).strip()
        md = {"dockercommit" : dockercommit}
    else:
        md = {}


    all_srs = []
    for i_run in xrange(args.start_run, n_runs):
        for (script_name,script_info) in script_dict.items():
            script_info = dict_update(default_settings, script_info)
            sr = ScriptRun(script_info,script_name,i_run, vmdir, md)
            all_srs.append(sr)

    if args.scripts_per_machine:
        scripts_per_machine = args.scripts_per_machine
    elif args.one_script_per_machine:
        scripts_per_machine = 1
    elif args.one_run_per_machine:
        scripts_per_machine = len(script_dict)

    assert_script_runs_different(all_srs)
    # make_bash_script(expt_name, config, srs, local=False, shutdown_instance=False, upload_result=False):
    bash_scripts = [make_bash_script(args.expt_name, config, srchunk, 
        cloud=args.cloud, 
        shutdown_instance = (not args.keep_instance) and args.cloud,
        upload = True) 
        for srchunk in chunkify(all_srs, scripts_per_machine)]

    if args.cloud:
        import boto3
        ec2 = boto3.resource("ec2")
        for bs in bash_scripts:
            print "*********************************************************"
            print bs
            print "*********************************************************"
            if not args.dry: ec2.create_instances(
                ImageId=config["aws"]["ami"],
                MinCount=1,
                MaxCount=1,
                UserData = bs,
                InstanceType = config["aws"]["instance_type"],
                SecurityGroups = config["aws"]["security_groups"],
                KeyName = config["aws"]["key_name"]
            )        
        print "%s started %i instances"%("(dry) " if args.dry else "", len(bash_scripts))
    else:
        for bs in bash_scripts:
            tmp = tempfile.NamedTemporaryFile(delete=False)
            print "writing script to",tmp.name
            tmp.file.write(bs)
            tmp.file.flush()
            maybe_call_and_print("/bin/bash %s"%tmp.name,args.dry)





if __name__ == "__main__":
    main()