# rllab

## Setup Instructions

- Install pip libraries:

  ```
  pip install joblib pyprind
  ```

- Install pygame:

  If using Anaconda, run the following:

  ```
  # linux
  conda install -c https://conda.binstar.org/tlatorre pygame
  # mac
  conda install -c https://conda.anaconda.org/quasiben pygame
  ```

  Otherwise, follow the official instructions.

- Build from source (required to run mjc / mjc2)
  ```
  mkdir build
  cd build
  cmake ../src
  make
  ```

  Make sure that you are linking against the correct Python. If using anaconda, this might give you trouble. The command that worked for me was like the following:
  ```
  CXXFLAGS="-I~/anaconda/include/python2.7" cmake -DPYTHON_PREFIX=~/anaconda -DPYTHON_LIBRARY=~/anaconda/lib/libpython2.7.dylib -DPYTHON_INCLUDE_DIR=~/anaconda/include -I~/anaconda/include/python2.7 ../src
  ```

  You also need to append `rllab/build/lib` to `PYTHONPATH`. If this succeeds, you should be able to run `python -c "import mjcpy"` and `python -c "import mjcpy2"` successfully.

  If you see an error `Segmentation fault: 11`, most likely you linked to the wrong Python. You can debug this by running:
  ```
  lldb -- python -c "import mjcpy"
  ```
  In the opened REPL, type `run`. It will break on the line that segfaults. Then type `bt`, which will give you a backtrace. You can also inspect the dynamically linked libraries by typing `image list`. Inspect in particular which `libpython2.7.dylib` it's using.

