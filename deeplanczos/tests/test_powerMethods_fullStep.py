import jax
import jax.random as jar
import jax.numpy as jnp
import jax.tree_util as jtu
from functools import partial
import pytest
from utils.BN_mlp import SimpleMLP_wBN

# @pytest.mark.slow
def test_power_method_step():
    from utils.BN_mlp import SimpleMLP_wBN, cross_entropy_loss
    from deep_lanczos import GenerateRandomVectors
    from deep_lanczos import dot_product
    from deep_lanczos import InitPowerMethodVector, ComputeHVP, initializeHVP
    from deep_lanczos import PowerMethodStep
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
    V_new, alpha = PowerMethodStep(loss, P1, iter(batch_iter), V, batch_parser=parser) 
    print(f'alpha = {alpha}')
    assert jtu.tree_structure(V) == jtu.tree_structure(V_new)


@pytest.mark.slow
def test_power_method():
    from utils.BN_mlp import SimpleMLP_wBN, cross_entropy_loss
    from deep_lanczos import GenerateRandomVectors
    from deep_lanczos import dot_product
    from deep_lanczos import InitPowerMethodVector, ComputeHVP, initializeHVP
    from deep_lanczos import PowerMethodStep
    from deep_lanczos import accumulateHVP
    from deep_lanczos import PowerMethodIterate
    from utils.batch import batch_iterator 
    from deep_lanczos.utils import check_shapes 

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

    parser = lambda B: (B['imgs'], B['labels'] )
    V = InitPowerMethodVector(rng, P1)
    assert check_shapes(V, P1)
    History = PowerMethodIterate(loss, P1, batch_iter, 30, V0=V, batch_parser=parser, show_pbar=True)
    print(jnp.array(History['alpha']))