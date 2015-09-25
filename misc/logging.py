import sys
import subprocess
import time
import os

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)

def colorize(string, color, bold=False, highlight = False):
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

MESSAGE_DEPTH = 0
class Message(object):
    def __init__(self, msg):
        self.msg = msg
    def __enter__(self):
        global MESSAGE_DEPTH #pylint: disable=W0603
        print colorize('\t'*MESSAGE_DEPTH + '=: ' + self.msg,'magenta')
        self.tstart = time.time()
        MESSAGE_DEPTH += 1
    def __exit__(self, etype, *args):
        global MESSAGE_DEPTH #pylint: disable=W0603
        MESSAGE_DEPTH -= 1
        maybe_exc = "" if etype is None else " (with exception)"
        print colorize('\t'*MESSAGE_DEPTH + "done%s in %.3f seconds"%(maybe_exc, time.time() - self.tstart), 'magenta')

def log(s):
    print s
    sys.stdout.flush()

def prefix_log(prefix):
    return lambda s: log(prefix + s)
