# hierctrl

## Setup Instructions

- Install pip libraries:

  ```
  pip install joblib pyprind
  ```

- Install tensorfuse:

  Grab `tensorfuse` from https://github.com/dementrock/tensorfuse. The recommended way to install is `python setup.py develop`. You may need to extend this library when certain operations are not supported.

- Install forked Lasagne:

  Grab `Lasagne` from https://github.com/dementrock/Lasagne. Again install using `python setup.py develop`.

- Install pygame:

  If using Anaconda, run the following:

  ```
  conda install -c https://conda.binstar.org/tlatorre pygame
  ```

  Otherwise, follow the official instructions.
