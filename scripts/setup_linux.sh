#!/bin/bash
# Make sure that pip is available
echo "Installing system dependencies"
echo "You will probably be asked for your sudo password."
sudo apt-get update
sudo apt-get install -y python-pip python-dev swig
sudo apt-get build-dep -y python-pygame
sudo apt-get build-dep -y python-scipy
# Make sure that virtualenv is available
hash virtualenv 2>/dev/null || {
  echo "Installing virtualenv"
  sudo pip install virtualenv
}

# Make sure that we're under the directory of the project
cd "$(dirname "$0")/.."
# Ensure .env

if [ ! -d ".env" ]; then
  virtualenv .env
fi
source .env/bin/activate
pip install -r requirements.txt
