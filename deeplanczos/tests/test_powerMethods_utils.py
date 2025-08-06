import jax
import jax.random as jar
import jax.numpy as jnp
import jax.tree_util as jtu
from functools import partial
import pytest
from utils.BN_mlp import SimpleMLP_wBN


def test_init():
    model = SimpleMLP_wBN([32, 32])
    sample_input = jnp.ones((1, 16))
    rng = jar.PRNGKey(0)
    par1 = model.init(rng, sample_input)
    assert "batch_stats" in par1.keys()


def test_init_loss():
    from utils.BN_mlp import SimpleMLP_wBN, cross_entropy_loss
    from deeplanczos import GenerateRandomVectors

    model = SimpleMLP_wBN([8, 8])
    rng_sample, rng_labels = jar.split(jar.PRNGKey(1994), 2)

    sample_input = jar.normal(rng_sample, (8, 16))
    sample_labels = jar.randint(rng_labels, (8,), minval=0, maxval=8)

    rng = jar.PRNGKey(0)
    par = model.init(rng, sample_input)

    rng_lanczos_vectors = jar.PRNGKey(10)
    W_uniform = GenerateRandomVectors(par['params'], rng_lanczos_vectors, 4, jar.uniform)
    W_gauss = GenerateRandomVectors(par['params'], rng_lanczos_vectors, 4)
    
    assert 0 == 0

# @pytest.mark.slow
def test_dot_product():
    from utils.BN_mlp import SimpleMLP_wBN, cross_entropy_loss
    from deeplanczos import GenerateRandomVectors
    from deeplanczos import dot_product
    from deeplanczos import InitPowerMethodVector
    from deeplanczos import orthogonlize

    model = SimpleMLP_wBN([8, 8])
    rng_sample, rng_labels = jar.split(jar.PRNGKey(1994), 2)

    sample_input = jar.normal(rng_sample, (8, 16))
    sample_labels = jar.randint(rng_labels, (8,), minval=0, maxval=8)
    rng1, rng2 = jar.split(jar.PRNGKey(0), 2) 

    par1 = model.init(rng1, sample_input)
    P1 = par1['params']
    par2 = model.init(rng2, sample_input)
    P2 = par2['params']
    v = InitPowerMethodVector(jar.PRNGKey(0), P1 )
    dot =  dot_product(P1, P2)
    P1 = orthogonlize(P1, P2)
    assert 0 == 0