import jax 
import flax.linen as nn 
from typing import Sequence
import jax.numpy as jnp 


class SimpleMLP_wBN(nn.Module):
    features: Sequence[int]
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Simple two-layer MLP with batch normalization.
        
        Args:
            x: Input array of shape (batch_size, input_dim)
            training: Whether in training mode (affects batch_norm behavior)
            
        Returns:
            Output array of shape (batch_size, features[-1])
        """
        x = nn.Dense(features=self.features[0])(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        
        x = nn.Dense(features=self.features[1])(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        
        return x
    
class SimpleMLP(nn.Module):
    features: Sequence[int]
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Simple two-layer MLP with batch normalization.
        
        Args:
            x: Input array of shape (batch_size, input_dim)
            training: Whether in training mode (affects batch_norm behavior)
            
        Returns:
            Output array of shape (batch_size, features[-1])
        """
        x = nn.Dense(features=self.features[0])(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        
        x = nn.Dense(features=self.features[1])(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        
        return x

def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    labels_onehot = jax.nn.one_hot(labels, num_classes=logits.shape[-1])
    log_softmax = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.sum(labels_onehot * log_softmax, axis=-1))