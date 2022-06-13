FROM nvidia/cuda:11.1.1-devel-ubuntu18.04

RUN apt-get update 

RUN apt-get install -y curl

# Install python3
RUN apt-get install -y python3.8
RUN apt-get install -y python3-distutils

RUN ln -sf /usr/bin/python3.8 /usr/bin/python 

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python get-pip.py

# install jax and jaxlib

RUN pip --no-cache-dir install --upgrade pip setuptools_rust

# Install ML Packages built with CUDA11 support
RUN pip install --upgrade jax jaxlib==0.3.7+cuda11.cudnn805 -f https://storage.googleapis.com/jax-releases/jax_releases.html

#git required to pip install from git repo's
RUN apt-get install -y git

#run requirements in order
#required because scipy needs numpy to already be installed. see https://stackoverflow.com/questions/51399515/docker-cannot-build-scipy.
COPY requirements.txt .

RUN while read module; do pip3 install $module; done < requirements.txt

RUN mkdir -p /home
RUN mkdir -p /home/app
WORKDIR /home/app
