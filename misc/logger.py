import sys
from misc.tabulate import tabulate

prefixes = []
prefix_str = ''
tabular = []

def push_prefix(prefix):
    prefixes.append(prefix)
    global prefix_str
    prefix_str = ''.join(prefixes)

def log(s):
    print prefix_str + s
    sys.stdout.flush()

def record_tabular(key, val):
    tabular.append((str(key), str(val)))

def dump_tabular():
    for line in tabulate(tabular).split('\n'):
        log(line)
    del tabular[:]

def pop_prefix():
    del prefixes[-1]
    global prefix_str
    prefix_str = ''.join(prefixes)
