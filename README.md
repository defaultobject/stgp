# Spatio-Temporal Gaussian Processes (`STGP`)

`stgp` is a Gaussian process library written in `Jax`.

## Installation

### CPU Setup

```
conda create -n stgp python=3.12
conda activate stgp
pip install -r requirements.txt
pip install -e .
conda install ipykernel nb_conda_kernels jupyter
```

## Docker

### Build the dockerfile [for mac]

construct a builder

```
docker buildx create --platform linux/arm64,linux/amd64 --driver=docker-container --name stgp_builder --use

docker buildx build --load -t defaultobject/stgp:latest -f dockerfile/stgp.Dockerfile .
```

## For Development

Setup precommit hooks and github actions

```
pip install pre-commit
pre-commit install
```

Docs are automatically generated from the scripts in `examples/` using `jupytext`. They can be generated with `make docs`.

# Cite this project

If you use STGP in academic work, please cite it as:

> Hamelijnck, O. STGP: Spatio-Temporal Gaussian Processes in JAX. GitHub repository, 2025.
https://github.com/defaultobject/stgp

```bibtex
@software{hamelijnck_stgp,
  author       = {Hamelijnck, O},
  title        = {STGP: Spatio-Temporal Gaussian Processes in JAX},
  year         = {2025},
  publisher    = {GitHub},
  url          = {https://github.com/defaultobject/stgp}
}
```
