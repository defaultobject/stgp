# Spatio-Temporal Gaussian Processes (`STGP`)

`stgp` is a Gaussian process library written in `Jax`.

## Installation

### m1 specific steps

it seems that it is easier to install tensorflow properly first and then figure out jax and stgp

```bash
 conda create -n stgp_lib_only python=3.9
 conda activate stgp_lib_only
 # set up tf for m1
 conda install -c apple tensorflow-deps
 pip install tensorflow-metal
 pip install tensorflow-macos

 # pip will complain but requried for tensorflow
 pip install numpy --upgrade

 pip install tensorflow_probability==0.23

 # for m1 jax -- for some reason need to specify most recent release?
 # must use conda-forge channel for m1
 conda install jax==0.4.8 -c conda-forge

 # now that tensorflow and jax is installed properly we can install stgp
 pip install -e . 
```

## Docker

### Build the dockerfile

construct a builder

```
docker buildx create --platform linux/arm64,linux/amd64 --driver=docker-container --name stgp_builder --use 


docker buildx build --load -t defaultobject/stgp:latest -f dockerfile/stgp.Dockerfile --cache-to type=local,dest=/Users/ohamelijnck/Documents/docker_cache --cache-from type=local,src=/Users/ohamelijnck/Documents/docker_cache .
```

## Table of contents

- [Implemented models](#implemented_models)

- [Examples](#examples)
  
  - [GP - Batch Form](#gp_batch_form)
  - [GP - VI](#gp_vi)
  - [GP - State Space](#gp_sde)

- [Sharp Bits](#sharp_bits)

## Install <a name="install"></a>

```bash
pip install -e stgp
```

## Implemented Models <a name="implemented_models"></a>

|                  | Batch Inference | MF VI   | Full rank VI | State space inference |
| ---------------- | --------------- | ------- | ------------ | --------------------- |
| GP               | &#9745;         | &#9745; | N/A          | &#9745;               |
| LMC              | &#9745;         | &#9745; |              | &#9746;               |
| Constrained LMC  | &#9745;         | &#9745; |              | &#9746;               |
| GPRN             | &#9745;         | &#9745; |              | &#9746;               |
| Constrained GPRN | &#9745;         | &#9745; |              | &#9746;               |

## Examples <a name="examples"></a>

### Toy Data

```python

```

### GP - Batch Form <a name="gp_batch_form"></a>

### GP - Variational Inference <a name="gp_vi"></a>

### GP - State Space <a name="gp_sde"></a>
