# Spatio-Temporal Gaussian Processes (`STGP`)

`stgp` is a Gaussian process library written in `Jax`.

## Installation

### m1 specific steps

```bash
 conda create -n stgp_lib_only python=3.10
 conda activate stgp_lib_only
 conda install -c apple tensorflow-deps
 pip install -e ".[m1]"
 pip install numpy --upgrade
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
