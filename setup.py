#!/usr/bin/env python3
from os.path import dirname, realpath
from setuptools import find_packages, setup


def _read_requirements_file():
    req_file_path = '%s/requirements.txt' % dirname(realpath(__file__))
    with open(req_file_path) as f:
        return [line.strip() for line in f]

setup(
    name='rllab',
    version='0.0.1-dev',
    packages=find_packages(),
    install_requires=_read_requirements_file(),
    package_data={
        '': ['*.pyx', '*.pxd', '*.pxi'],
    })
