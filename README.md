# Spatio-Temporal Gaussian Processes (`STGP`)

`stgp` is a Gaussian process library written in `Jax`.

## Installation

### m1 specific steps

```bash
conda create -n stgp python=3.11
conda activate stgp
pip install tensorflow-metal
pip install tensorflow_probability==0.23
conda install -c conda-forge jax==0.4.25 jaxlib==0.4.25

pip install jax
pip install tensorflow tensorflow_probability tf_keras
pip install -r requirements.txt

conda install python-graphviz
pip install networkx

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
