#!/bin/bash

if [ "$(uname)" == "Darwin" ]; then
    mujoco_file="libmujoco.dylib"
    glfw_file="libglfw.3.dylib"
    zip_file="mjpro122_osx.zip"
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    mujoco_file="libmujoco.so"
    glfw_file="libglfw.so.3"
    zip_file="mjpro122_linux.zip"
fi

if [ ! -f vendor/mujoco122/$mujoco_file ]; then
    read -e -p "Please enter the path to the mujoco zip file [$zip_file]:" path
    path=${path:-$zip_file} 
    eval path=$path
    if [ ! -f $path ]; then
        echo "No file found at $path"
        exit 0
    fi
    rm -r /tmp/mujoco
    dir=`mktemp -d /tmp/mujoco`
    unzip $path -d $dir
    if [ ! -f $dir/mjpro/$mujoco_file ]; then
        echo "mjpro/$mujoco_file not found. Make sure you have the correct file (most likely named $zip_file)"
        exit 0
    fi
    if [ ! -f $dir/mjpro/$glfw_file ]; then
        echo "mjpro/$glfw_file not found. Make sure you have the correct file (most likely named $zip_file)"
        exit 0
    fi

    mkdir -p vendor/mujoco122
    cp $dir/mjpro/$mujoco_file vendor/mujoco122/
    cp $dir/mjpro/$glfw_file vendor/mujoco122/
fi

if [ ! -f vendor/mujoco122/mjkey.txt ]; then
    read -e -p "Please enter the path to the mujoco license file [mjkey.txt]:" path
    path=${path:-mjkey.txt}
    eval path=$path
    if [ ! -f $path ]; then
        echo "No file found at $path"
        exit 0
    fi
    cp $path vendor/mujoco122/mjkey.txt
fi

echo "Mujoco has been set up!"
