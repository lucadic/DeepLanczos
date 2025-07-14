import jax
import jax.random as jar
import jax.numpy as jnp
import jax.tree_util as jtu
from functools import partial
import pytest
from utils.BN_mlp import SimpleMLP_wBN

# @pytest.mark.slow
def test_HVP():
    from utils.BN_mlp import SimpleMLP_wBN, cross_entropy_loss
    from deep_lanczos import GenerateRandomVectors
    from deep_lanczos import dot_product
    from deep_lanczos import InitPowerMethodVector, ComputeHVP
    from utils.batch import batch_iterator

    model = SimpleMLP_wBN([5, 5])
    rng_sample, rng_labels = jar.split(jar.PRNGKey(1994), 2)

    X = jar.normal(rng_sample, (128, 4))
    Y = jar.randint(rng_labels, (128,), minval=0, maxval=8)
    batch_iter = batch_iterator(X, Y)
    sample_input = next(iter(batch_iter))['imgs']

    rng = jar.PRNGKey(0)
    par = model.init(rng, sample_input)
    P1 = par['params']
    @jax.jit
    def loss(params, x, yhat):
        logits = model.apply(
        {"params": params, "batch_stats": par['batch_stats']},
        x,
        training=False,
        )
        return cross_entropy_loss(logits, yhat)
    V = InitPowerMethodVector(rng, P1) 
    parser = lambda B: (B['imgs'], B['labels'] )
    batch = next(iter(batch_iter)) 
    HVP = ComputeHVP(loss, P1, batch, V, batch_parser=parser) 
       
    assert jtu.tree_structure(P1) == jtu.tree_structure(HVP)

    shapes_check = jtu.tree_map(lambda x, y : x.shape == y.shape, P1, HVP) 
    assert jtu.tree_reduce(lambda s1, s2 : s1 and s2, shapes_check, initializer=True)
    print(HVP)

# @pytest.mark.slow
def test_HVP_cycle():
    from utils.BN_mlp import SimpleMLP_wBN, cross_entropy_loss
    from deep_lanczos import GenerateRandomVectors
    from deep_lanczos import dot_product
    from deep_lanczos import InitPowerMethodVector, ComputeHVP, initializeHVP
    from deep_lanczos import accumulateHVP
    from utils.batch import batch_iterator 

    model = SimpleMLP_wBN([5, 5])
    rng_sample, rng_labels = jar.split(jar.PRNGKey(1994), 2)

    X = jar.normal(rng_sample, (128, 4))
    Y = jar.randint(rng_labels, (128,), minval=0, maxval=8)
    batch_iter = batch_iterator(X, Y)
    sample_input = next(iter(batch_iter))['imgs']

    rng = jar.PRNGKey(0)
    par = model.init(rng, sample_input)
    P1 = par['params']
    @jax.jit
    def loss(params, x, yhat):
        logits = model.apply(
        {"params": params, "batch_stats": par['batch_stats']},
        x,
        training=False,
        )
        return cross_entropy_loss(logits, yhat)
    batch = next(iter(batch_iter)) 
    

    parser = lambda B: (B['imgs'], B['labels'] )
    V = InitPowerMethodVector(rng, P1)
    HVP = initializeHVP(P1)
    shapes_check = jtu.tree_map(lambda x, y : x.shape == y.shape, P1, HVP) 
    assert  jtu.tree_reduce(lambda s1, s2 : s1 and s2, shapes_check, initializer=True)
    
    for batch in iter(batch_iter):  
        HVP_contribution = ComputeHVP(loss, P1, batch, V, batch_parser=parser) 
        HVP = accumulateHVP(HVP_contribution, HVP)
        print('________________________________________')
        print(HVP)

