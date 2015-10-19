import sys

prefixes = []
prefix_str = ''

def push_prefix(prefix):
    prefixes.append(prefix)
    global prefix_str
    prefix_str = ''.join(prefixes)

def log(s):
    print prefix_str + s
    sys.stdout.flush()

def pop_prefix():
    del prefixes[-1]
    global prefix_str
    prefix_str = ''.join(prefixes)
