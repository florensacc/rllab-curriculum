#!/bin/sh
parent_path=$( cd "$(dirname "${BASH_SOURCE}")" ; pwd -P )
gcc -E -CC $parent_path/../../../vendor/mujoco/1_20/osx/mujoco.h > /tmp/code_gen_mujoco.h
ruby $parent_path/codegen.rb /tmp/code_gen_mujoco.h > $parent_path/mjtypes.py
