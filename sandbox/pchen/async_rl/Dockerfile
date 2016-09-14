FROM ubuntu:16.04

# ========== Anaconda ==========
# https://github.com/ContinuumIO/docker-images/blob/master/anaconda3/Dockerfile

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/archive/Anaconda3-4.1.1-Linux-x86_64.sh -O ~/anaconda.sh && \
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
CMD [ "/bin/bash" ]

# ========== Other python dependencies ===============
RUN pip install chainer

RUN wget https://pypi.python.org/packages/a9/70/a0e9c2f2d369808843d4b49c9f92f247ce54e50a684cc715d818e866ad30/cached_property-1.3.0-py2.py3-none-any.whl#md5=6eed185585787079dd52024a071757f7
RUN pip install cached_property-1.3.0-py2.py3-none-any.whl

# cmake
RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:george-edison55/cmake-3.x
RUN apt-get update -y
RUN apt-get install cmake -y
RUN apt-get install build-essential -y

# ALE
RUN apt-get install git
RUN git clone https://github.com/mgbellemare/Arcade-Learning-Environment.git
RUN apt-get install libsdl1.2-dev libsdl-gfx1.2-dev libsdl-image1.2-dev -y
RUN cd Arcade-Learning-Environment && mkdir build && cd build && \
    cmake -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=ON .. && \
    make -j 4 && \
    cd .. && pip install .

RUN conda install -c https://conda.binstar.org/menpo opencv3

RUN pip install joblib
RUN apt-get install gtk3.0 -y

# # ========== Add codebase stub ==========
CMD mkdir /root/code
ENV PYTHONPATH /root/code/async_rl:$PYTHONPATH
ENV BASH_ENV /root/.bashrc
WORKDIR /root/code
ADD . /root/code/

