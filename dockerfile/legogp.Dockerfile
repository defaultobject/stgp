FROM nvidia/cuda:10.2-devel-ubuntu18.04

# Install python3
RUN apt update && apt install -y python3-pip

RUN ln -sf /usr/bin/python3 /usr/bin/python && \
ln -sf /usr/bin/pip3 /usr/bin/pip

RUN pip --no-cache-dir install --upgrade pip setuptools_rust

# Install ML Packages built with CUDA11 support
RUN ln -s /usr/lib/cuda /usr/local/cuda-10.2
RUN pip --no-cache-dir install --upgrade jax[cuda102] -f https://storage.googleapis.com/jax-releases/jax_releases.html

#git required to pip install from git repo's
RUN apt-get install -y git

#run requirements in order
#required because scipy needs numpy to already be installed. see https://stackoverflow.com/questions/51399515/docker-cannot-build-scipy.
COPY requirements.txt .

RUN while read module; do pip3 install $module; done < requirements.txt

RUN mkdir -p /home
RUN mkdir -p /home/app
WORKDIR /home/app
