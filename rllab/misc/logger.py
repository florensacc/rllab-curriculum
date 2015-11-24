from __future__ import print_function
from rllab.misc.tabulate import tabulate
from rllab.misc.console import mkdir_p
import os
import os.path as osp
import sys
import datetime
import dateutil.tz
import csv
import joblib

_prefixes = []
_prefix_str = ''
_tabular = []

_text_outputs = []
_tabular_outputs = []

_text_fds = {}
_tabular_fds = {}
_tabular_header_written = set()

_snapshot_dir = None
_snapshot_mode = 'all'


def _add_output(file_name, arr, fds, mode='a'):
    if file_name not in arr:
        mkdir_p(os.path.dirname(file_name))
        arr.append(file_name)
        fds[file_name] = open(file_name, 'a')


def _remove_output(file_name, arr, fds):
    if file_name in arr:
        fds[file_name].close()
        del fds[file_name]
        arr.remove(file_name)


def push_prefix(prefix):
    _prefixes.append(prefix)
    global _prefix_str
    _prefix_str = ''.join(_prefixes)


def add_text_output(file_name):
    _add_output(file_name, _text_outputs, _text_fds, mode='a')


def remove_text_output(file_name):
    _remove_output(file_name, _text_outputs, _text_fds)


def add_tabular_output(file_name):
    _add_output(file_name, _tabular_outputs, _tabular_fds, mode='wb')


def remove_tabular_output(file_name):
    _tabular_header_written.remove(_tabular_fds[file_name])
    _remove_output(file_name, _tabular_outputs, _tabular_fds)


def set_snapshot_dir(dir_name):
    global _snapshot_dir
    _snapshot_dir = dir_name


def get_snapshot_dir():
    return _snapshot_dir


def get_snapshot_mode():
    return _snapshot_mode


def set_snapshot_mode(mode):
    global _snapshot_mode
    _snapshot_mode = mode


def log(s, with_prefix=True, with_timestamp=True):
    out = s
    if with_prefix:
        out = _prefix_str + out
    if with_timestamp:
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f %Z')
        out = "%s | %s" % (timestamp, out)
    # Also log to stdout
    print(out)
    for fd in _text_fds.values():
        fd.write(out + '\n')
        fd.flush()
    sys.stdout.flush()


def record_tabular(key, val):
    _tabular.append((str(key), str(val)))


def dump_tabular(*args, **kwargs):
    for line in tabulate(_tabular).split('\n'):
        log(line, *args, **kwargs)
    tabular_dict = dict(_tabular)
    # Also write to the csv files
    # This assumes that the keys in each iteration won't change!
    for tabular_fd in _tabular_fds.values():
        writer = csv.DictWriter(tabular_fd, fieldnames=tabular_dict.keys())
        if tabular_fd not in _tabular_header_written:
            writer.writeheader()
            _tabular_header_written.add(tabular_fd)
        writer.writerow(tabular_dict)
    del _tabular[:]


def pop_prefix():
    del _prefixes[-1]
    global _prefix_str
    _prefix_str = ''.join(_prefixes)


def save_itr_params(itr, params):
    if _snapshot_dir:
        if _snapshot_mode == 'all':
            file_name = osp.join(_snapshot_dir, 'itr_%d.pkl' % itr)
            joblib.dump(params, file_name, compress=3)
        elif _snapshot_mode == 'last':
            # override previous params
            file_name = osp.join(_snapshot_dir, 'params.pkl')
            joblib.dump(params, file_name, compress=3)
        elif _snapshot_mode == 'none':
            pass
        else:
            raise NotImplementedError
