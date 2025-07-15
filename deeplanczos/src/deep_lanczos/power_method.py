import jax.numpy as jnp
import jax.tree_util as jtu 
import jax.random as jar
from tqdm import tqdm 
import jax 
from .utils import split_rng_for_leaves
from .utils import accumulateHVP
from .utils import check_shapes
def mdot(p1, p2): 
    return (p1 * p2).sum()




def dot_product(P1, P2): 
    dot_product = jtu.tree_map(mdot, P1, P2)
    return jtu.tree_reduce(lambda S, cdot: S + cdot, dot_product, initializer=0)


def orthogonlize(P, Q): 
    cdot = dot_product(P, Q) 
    Q_norm = dot_product(Q, Q)
    return jtu.tree_map(lambda p, q : p - q * cdot/(Q_norm), P, Q) 

def norm2(P): 
    return dot_product(P,P)

def normalize(P): 
    P_norm = dot_product(P, P)**0.5
    return jtu.tree_map(lambda p: p/P_norm, P)

def InitPowerMethodVector(rng, P, rng_gen = jar.normal): 
    def generate_entry(rng, p): 
        return rng_gen(rng, p.shape)
    
    rngs = split_rng_for_leaves(rng, P)
    return normalize(jtu.tree_map(generate_entry, rngs, P) )


def initializeHVP(par): 
    return jtu.tree_map(lambda x : jnp.zeros(x.shape), par)

def standard_parser(batch): 
     return (batch[0], batch[1]) 

def ComputeHVP (loss, par, batch, V, batch_parser=standard_parser):
    x, yhat = batch_parser(batch) 
    grad = jax.grad(lambda p : loss(p, x, yhat), argnums=0) 
    HVP = jax.jvp(grad, (par,), (V,) )[1]
    return HVP 


def PowerMethodStep(loss, par, batch_iterator, V, **kwargs):
    HVP = initializeHVP(par)
    show_pbar = kwargs.get('show_pbar', False)
    batch_parser = kwargs.get('batch_parser', standard_parser)
    momentum = kwargs.get('momentum', 0)
    count = 0
    iterator = batch_iterator if not show_pbar else tqdm(batch_iterator)
    for batch in iterator: 
        HVP_contribution = ComputeHVP(loss, par, batch, V, batch_parser=batch_parser) 
        HVP = accumulateHVP(HVP_contribution, HVP)
        count += 1
    HVP = jtu.tree_map(lambda x: x/count, HVP) 
    alpha = dot_product(HVP, HVP)/ dot_product(V, V)  
    V_new = normalize(HVP)
    V_new = jtu.tree_map(lambda x, y : momentum*x + (1-momentum) * y, V, V_new)
    return V_new, alpha  

def PowerMethodIterate(loss, par, batch_iterable, num_iterations, V0=None, **kwargs): 
    
    if V0 == None:
        rng = jar.PRNGKey(0)
        V0 = InitPowerMethodVector(rng, par)
        
    if jtu.tree_structure(V0) != jtu.tree_structure(par): 
        raise ValueError("V0 and par must have the same tree structure, or V0 should be None")
    
    if check_shapes(V0, par) is not True: 
        raise ValueError(f"The leaves of V0 and par must have the same shapes") 
    
    
    V = V0.copy() 
    History = {'alpha' : [], 'iteration': [], 'q' : []}
    for iteration in range(num_iterations): 
       batch_iterator = iter(batch_iterable)  # Reset iterator for each iteration
       V_new, alpha = PowerMethodStep(loss, par, batch_iterator, V, **kwargs) 
       q = dot_product(V, V_new)/( norm2(V)  * norm2(V_new))**0.5 
       V = V_new 
       
       History['alpha'].append(alpha) 
       History['iteration'].append(iteration)
       History['q'].append(q)

    return History

