FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

# declare the image name
ENV IMG_NAME=11.2.2-cudnn8-devel-ubuntu20.04 \
    # declare what jaxlib tag to use
    # if a CI/CD system is expected to pass in these arguments
    # the dockerfile should be modified accordingly
    JAXLIB_VERSION=0.1.65

# install python3-pip
RUN apt update && apt install python3-pip -y


# install dependencies via pip
RUN pip3 install numpy scipy six wheel jaxlib==${JAXLIB_VERSION}+cuda112 -f https://storage.googleapis.com/jax-releases/jax_releases.html

#run requirements in order
#required because scipy needs numpy to already be installed. see https://stackoverflow.com/questions/51399515/docker-cannot-build-scipy.
COPY requirements.txt .
#hack for now untill model_log can be pip installed
RUN while read module; do pip3 install $module; done < requirements.txt

RUN apt-get install -y git

RUN alias python=python3

RUN ln -s /usr/bin/python3 /usr/bin/python 

# Create a working directory
RUN mkdir /app
WORKDIR /app
