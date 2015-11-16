import sys
from rllab.misc.tabulate import tabulate
from rllab.misc.console import mkdir_p
import os
import pytz
import datetime
import dateutil.tz

prefixes = []
prefix_str = ''
tabular = []
file_outputs = []
fds = {}

def push_prefix(prefix):
    prefixes.append(prefix)
    global prefix_str
    prefix_str = ''.join(prefixes)

def add_file_output(file_name):
    if file_name not in file_outputs:
        mkdir_p(os.path.dirname(file_name))
        file_outputs.append(file_name)
        fds[file_name] = open(file_name, 'a')

def remove_file_output(file_name):
    if file_name in file_outputs:
        fds[file_name].close()
        del fds[file_name]
        file_outputs.remove(file_name)

def log(s, with_prefix=True, with_timestamp=True):
    out = s
    if with_prefix:
        out = prefix_str + out
    if with_timestamp:
        timestamp = datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%Y-%m-%d %H:%M:%S.%f %Z')
        out = "%s | %s" % (timestamp, out)
    print out
    for fd in fds.values():
        fd.write(out + '\n')
        fd.flush()
    sys.stdout.flush()

def record_tabular(key, val):
    #global tabular
    #tabular = filter(lambda x: x[0] != key, tabular)
    tabular.append((str(key), str(val)))

def dump_tabular(*args, **kwargs):
    for line in tabulate(tabular).split('\n'):
        log(line, *args, **kwargs)
    del tabular[:]

def pop_prefix():
    del prefixes[-1]
    global prefix_str
    prefix_str = ''.join(prefixes)
