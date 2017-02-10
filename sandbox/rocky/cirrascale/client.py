import datetime
import logging
import sys
import urllib.parse

import requests
import requests.exceptions

logger = logging.getLogger(__name__)

_HOST = 'http://gpu-monitor-api.sci.openai.org'
_STATUS_ENDPOINT = '/gpu/status'
_RESERVE_ENDPOINT = '/gpu/reserve'
_RELEASE_ENDPOINT = '/gpu/release'


class GPU(object):
    def __init__(self, data):
        self.id = data['id']
        self.processes = [GPUProcess(proc_data)
                          for proc_data in data['processes']]
        self.reserved = data.get('reserved')
        self.force_available = None

        (self.host, _, self.index) = self.id.partition(':')

    @property
    def available(self):
        if self.force_available is not None:
            return self.force_available
        # TODO: take into account reserved GPUs
        return len(self.processes) == 0 and not self.reserved

    def __str__(self):
        return 'GPU(id=%s)' % (self.id)

    def __repr__(self):
        return str(self)


class GPUProcess(object):
    _data_fields = ('cmdline', 'loginuid', 'name', 'start_time_s',
                    'used_memory', 'username')

    def __init__(self, data):
        for field in self._data_fields:
            setattr(self, field, data.get(field))

        if self.start_time_s:
            self.start_time = datetime.datetime.fromtimestamp(self.start_time_s)
        else:
            self.start_time = None
        self.used_memory_h = _sizeof_fmt(self.used_memory)


class ClientConnectionError(Exception):
    def __init__(self, message, orig_error, exc_info):
        super(ClientConnectionError, self).__init__(message)
        self.orig_error = orig_error
        self.exc_info = exc_info


def _make_request(path, method, **kwargs):
    url = urllib.parse.urljoin(_HOST, path)
    logger.debug('%s %s, kwargs: %s', method, url, kwargs)
    try:
        return requests.request(method, url, **kwargs)
    except requests.exceptions.ConnectionError as e:
        msg = ('Request %s %s failed to connect. '
               'Are you connected to the VPN?') % (method, path)
        raise ClientConnectionError(msg, e, sys.exc_info())


def get_gpu_status():
    """
    returns a dictionary of the form
    {
        "10.cirrascale.sci.openai-tech.com": GPU object,
        ...
    }
    """
    resp = _make_request(_STATUS_ENDPOINT, 'GET')
    raw_data = resp.json()

    results = {}

    for gpu_data in raw_data['data']:
        gpu = GPU(gpu_data)
        results.setdefault(gpu.host, []).append(gpu)

    for host, gpu_list in results.items():
        results[host] = sorted(gpu_list, key=lambda gpu: gpu.id)

    return results


def reserve_gpus(username, gpu_ids, ttl):
    """
    marks a set of GPUs as reserved.

    params:
    username - string
    gpu_ids - list of strings
    ttl - int. time in seconds.
    """
    logger.debug('Reserving %s', gpu_ids)
    data = {
        'username': username,
        'gpu_ids': gpu_ids,
        'ttl': ttl
    }
    resp = _make_request(_RESERVE_ENDPOINT, 'POST', json=data)
    resp.raise_for_status()
    return resp.json()


def release_gpus(gpu_ids):
    """
    marks a set of GPUs as available.

    params:
    gpu_ids - list of strings
    """
    logger.debug('Releasing %s', gpu_ids)
    data = {
        'gpu_ids': gpu_ids
    }
    resp = _make_request(_RELEASE_ENDPOINT, 'POST', json=data)
    resp.raise_for_status()
    return resp.json()


def _sizeof_fmt(num, suffix='B'):
    """
    source http://stackoverflow.com/questions/1094841/reusable-library-to-get-human-readable-version-of-file-size
    """
    for unit in ('', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi'):
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)
