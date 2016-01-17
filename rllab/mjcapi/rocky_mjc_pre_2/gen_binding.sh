#!/bin/sh
parent_path=$( cd "$(dirname "${BASH_SOURCE}")" ; pwd -P )
project_root=$parent_path/../../..
mjc2_path=$project_root/src/mjc2
mjc2_source_path=$mjc2_path/source
cat $mjc2_source_path/engine/engine_typedef.h \
    $mjc2_source_path/visual/visual_typedef.h \
    > /tmp/code_gen_mujoco.h
ruby $parent_path/codegen.rb /tmp/code_gen_mujoco.h > $parent_path/mjtypes.py
