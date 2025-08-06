Lanczos Algorithm Implementation in JAX

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
        y_batch: Target data batch
    Returns:
        float: Loss value
    """
    return ...

# Create Lanczos instance
lanczos = deeplanczos.PowerMethod(model, loss_fn)

# Compute largest eigenvalue and eigenvector
eigenvalue, eigenvector = lanczos.compute()
```


## License
MIT License
