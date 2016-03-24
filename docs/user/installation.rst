.. _installation:


============
Installation
============

Preparation
===========

You need to edit your :code:`PYTHONPATH` to include the rllab directory:

.. code-block:: bash

    export PYTHONPATH=path_to_rllab:$PYTHONPATH

Express Install
===============

The fastest way to set up dependencies for rllab is via running the setup script.

- On Linux, run the following:

.. code-block:: bash

    ./script/setup_linux.sh

- On Mac OS X, run the following:

.. code-block:: bash

    ./script/setup_osx.sh

The script sets up a virtual environment. To start using it, run the following:

.. code-block:: bash

    source .env/bin/activate


Optionally, if you would like to run experiments that depends on the Mujoco environment, you can set it up by running the following command:

.. code-block:: bash

    ./script/setup_mujoco.sh

and follow the instructions. You need to have the zip file for Mujoco v1.22 and the license file ready.



Manual Install
==============

Python + pip
------------

RLLab currently requires Python 2.7.4 to run. Please install Python via
the package manager of your operating system if it is not included already.

Python includes ``pip`` for installing additional modules that are not shipped
with your operating system, or shipped in an old version, and we will make use
of it below. You should set up a `virtual environment
<http://www.dabapps.com/blog/introduction-to-pip-and-virtualenv-python/>`_
like below:

.. code-block:: bash

    virtualenv --no-site-packages .env
    source .env/bin/activate

System dependencies for pygame
------------------------------

A few MDPs in RLLab are implemented using Box2D, which uses pygame for visualization.
It requires a few system dependencies to be installed first.

On Linux, run the following:

.. code-block:: bash

  sudo apt-get install swig
  sudo apt-get build-dep python-pygame

On Mac OS X, run the following:

.. code-block:: bash

  brew install swig sdl sdl_image sdl_mixer sdl_ttf portmidi

System dependencies for scipy
-----------------------------

This step is only needed under Linux:

.. code-block:: bash

  sudo apt-get build-dep python-scipy

Install Python modules
----------------------

.. code-block:: bash

  pip install -r requirements.txt
