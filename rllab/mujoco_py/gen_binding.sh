#!/bin/sh
parent_path=$( cd "$(dirname "${BASH_SOURCE}")" ; pwd -P )

rm /tmp/code_gen_mujoco.h
cat $parent_path/../../vendor/mujoco/mjdata.h >> /tmp/code_gen_mujoco.h
cat $parent_path/../../vendor/mujoco/mjmodel.h >> /tmp/code_gen_mujoco.h
cat $parent_path/../../vendor/mujoco/mjrender.h >> /tmp/code_gen_mujoco.h
cat $parent_path/../../vendor/mujoco/mjvisualize.h >> /tmp/code_gen_mujoco.h
# cat $parent_path/../../vendor/mujoco/mujoco.h >> /tmp/code_gen_mujoco.h
# gcc -E -CC $parent_path/../../vendor/mujoco/mujoco.h > /tmp/code_gen_mujoco.h
# gcc -E -CC $parent_path/../../vendor/mujoco/mujoco.h > /tmp/code_gen_mujoco.h
ruby $parent_path/codegen.rb /tmp/code_gen_mujoco.h $parent_path/../../vendor/mujoco/mjxmacro.h > $parent_path/mjtypes.py
