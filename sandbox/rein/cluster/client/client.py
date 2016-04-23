import httplib2
import urllib
import sys

# rllab-q server ip and port
IP = 'ec2-54-183-193-229.us-west-1.compute.amazonaws.com'
PORT = '8888'


def exec_command(rllab_q_address=None):
    h = httplib2.Http('.cache')
#     data = {'container_name': 'rein/rllab',
#             'expt_name': 'test',
#             'command': 'python /home/ubuntu/workspace/rllab/scripts/run_experiment.py'}
    data = {'container_name': 'perl',
            'expt_name': 'test',
            'command': 'perl -Mbignum=bpi -wle print bpi(2000)'}
    body = urllib.urlencode(data)
    resp, content = h.request(rllab_q_address,
                              "POST",
                              body)
    print(resp, content)

if __name__ == "__main__":
    address = 'http://' + IP + ':' + PORT
    exec_command(rllab_q_address=address)
