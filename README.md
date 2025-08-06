# Lanczos Algorithm Implementation in JAX

## Overview
This implementation provides a JAX-based power methods algorithm to compute the largest eigenvalues with the corresponding eigenvector for the Hessian of loss of deep networks

## Current features
- Pure JAX implementation for GPU/TPU acceleration
- Matrix-free implementation (requires only matrix-vector products)
- Automatic batch accumulation
- Power Method to compute the largest eigenvalue 

## Future features
- Implement full Lanczos algorithm for multiple eigenvalues
- Add support for distributed computing

## Installation
```bash
pip install deeplanczos
```

## Usage
```python
import jax
import deeplanczos

# Initialize your model and loss function
model = ...
def loss_fn(parameters, x_batch, y_batch) -> float:
    """Loss function for the model.
    Args:
        parameters: Model parameters
        x_batch: Input data batch
        y_batch: Target 
    Returns:
        float: Loss value
    """
    return ...

```
define a parser that define how to  parse x, y from a batch, for example if the batch is in the format batch = x, y
```python
parser = lambda x : (x[0], x[1])
```
while if the batch is  batch = {'imgs' : img, 'labels' : labels}
```python
parser = lambda batch : (batch['imgs'], batch['labels'] )
```
Given the PyTree of the model's parameters $P$, 
```python
from deeplanczos import InitPowerMethodVector
from deeplanczos import PowerMethodIterate

V = InitPowerMethodVector(rng, P)
num_iteration = 30
V, History = PowerMethodIterate(loss, P, batch_iterable, num_iteraion, V0=V, batch_parser=parser, show_pbar=True)
```
## Output

The function returns two values:
- `V`: The final estimate of the eigenvector corresponding to the largest eigenvalue
- `History`: A dictionary containing convergence metrics with the following keys:
    - `alpha`: Array of squared maximum eigenvalue estimates at each iteration
    - `q`: Array of overlaps between consecutive eigenvector estimates
    - `iteration`: Array of iteration step indices

These outputs allow you to monitor the convergence of the power method algorithm and analyze the spectrum of your neural network's Hessian.
## License
MIT License
