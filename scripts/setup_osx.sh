#!/bin/bash
# Make sure that pip is available
hash brew 2>/dev/null || {
    echo "Please install homebrew before continuing. You can use the following command to install:"
    echo "/usr/bin/ruby -e \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)\""
    exit 0
}

echo "Installing system dependencies"
echo "You will probably be asked for your sudo password."

sudo easy_install pip
brew install swig sdl sdl_image sdl_mixer sdl_ttf portmidi
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
