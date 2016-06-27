FROM ubuntu:16.04

# ========== Anaconda ==========
# https://github.com/ContinuumIO/docker-images/blob/master/anaconda/Dockerfile

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/archive/Anaconda2-4.0.0-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh

RUN apt-get install -y curl grep sed dpkg && \
    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean

ENV PATH /opt/conda/bin:$PATH

ENTRYPOINT [ "/usr/bin/tini", "--" ]

# ========== Special Deps ==========
CMD mkdir /root/code
# ALE requires zlib, make, unzip to build
RUN apt-get -y install zlib1g-dev make cmake unzip
# MUJOCO requires graphics stuff (Why?)
RUN apt-get -y build-dep glfw
RUN apt-get -y install libxrandr2 libxinerama-dev libxi6 libxcursor-dev
# For Gym video
RUN apt-get -y install libav-tools

RUN mkdir ~/.mujoco && curl -k https://openai-public.s3-us-west-2.amazonaws.com/mujoco/openai-linux-iegh3ahh.tar.gz | tar xz -C ~/.mujoco/

RUN pip install \
    joblib \
    mako \
    ipywidgets \
    flask \
    https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.9.0-cp27-none-linux_x86_64.whl \
    pip \
    atari-py \
    pyprind \
    ipdb \
    PyOpenGL \
    msgpack-python \
    mujoco_py \
    git+https://github.com/Theano/Theano.git@adfe319ce6b781083d8dc3200fb4481b00853791#egg=Theano \
    git+https://github.com/Lasagne/Lasagne.git@19b9d958507e7300e355d7cd1a6bbff2c7726c07#egg=Lasagne \
    git+https://github.com/plotly/plotly.py.git@2594076e29584ede2d09f2aa40a8a195b3f3fc66#egg=plotly \
    awscli \
    git+https://github.com/openai/gym.git

# # ========== Add codebase stub ==========

ENV PYTHONPATH /root/code/rllab:$PYTHONPATH
ENV PATH /opt/conda/envs/rllab/bin:$PATH
WORKDIR /root/code
