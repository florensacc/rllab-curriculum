import os
import argparse
from rllab.misc.instrument import query_yes_no
from rllab.misc.console import colorize

parser = argparse.ArgumentParser()
parser.add_argument('keyword',type=str, default='????????',nargs='?')
args = parser.parse_args()

log_file = "kube_pods.log"
os.system("kubectl get pods -l owner=hrtang > %s"%(log_file))
with open (log_file) as f:
    lines = f.readlines() 
    for i,line in enumerate(lines):
        if i > 0: # skip the title line
            pod_name = line.split(' ')[0]
            if args.keyword in pod_name:
                question = "Delete %s ?"%(pod_name)
                confirm_deletion = query_yes_no(question)
                if confirm_deletion:
                    os.system("kubectl delete pod %s"%(pod_name)) 
                    print(colorize("deleted...",color="red"))
                else:
                    print(colorize("aborted...",color="yellow"))
                
