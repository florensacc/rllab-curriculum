#!/bin/sh
parent_path=$( cd "$(dirname "${BASH_SOURCE}")" ; pwd -P )
project_root=$parent_path/../../..
mjc_path=$project_root/src/mjc
mjc_source_path=$mjc_path/core
cat $mjc_source_path/mj_engine.h \
    $mjc_source_path/mj_type.h \
    > /tmp/code_gen_mujoco.h
ruby $parent_path/codegen.rb /tmp/code_gen_mujoco.h > $parent_path/mjtypes.py
