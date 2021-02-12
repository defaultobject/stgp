# GPJAX

## Table of contents

 - [Implemented models](#implemented_models)

 - [Examples](#examples)

    - [GP - Batch Form](#gp_batch_form)
    - [GP - VI](#gp_vi)
    - [GP - State Space](#gp_sde)

 - [Sharp Bits](#sharp_bits)




## Install <a name="install"></a>

```bash
pip install -e gpjax
```



## Implemented Models <a name="implemented_models"></a>

|                  | Batch Inference | MF VI   | Full rank VI | State space inference |
| ---------------- | --------------- | ------- | ------------ | --------------------- |
| GP               | &#9745;         | &#9745; | N/A          | &#9745;               |
| LMC              | &#9745;         | &#9745; |              | &#9746;               |
| Constrained LMC  | &#9745;         | &#9745; |              | &#9746;               |
| GPRN             | &#9745;         | &#9745; |              | &#9746;               |
| Constrained GPRN | &#9745;         | &#9745; |              | &#9746;               |



##  Examples <a name="examples"></a>

### Toy Data

```python

```



### GP - Batch Form <a name="gp_batch_form"></a>

 

### GP - Variational Inference <a name="gp_vi"></a>



### GP - State Space <a name="gp_sde"></a>



## Sharp Bits <a name="sharp_bits"></a>

 

### Defining automatically registered classes

To automatically register a class as a jit'able instance simply extend `Module` :

```python
class Model(Module):
    def __init__(
        self, 
        <keywargs>
    ) -> None:
      <...>
```



For this to work all arguments to the `__init__` function but be keyword arguments and every argument must be stored as a property with the same name. The object must be able to be 'cloned' through the init arguments.

A helper method within `Module` is provided which automatically saves the arguments to properties:

```python
  #save __init__ arguments as properties of this object
  self.save_inputs_to_properties(locals())
```



