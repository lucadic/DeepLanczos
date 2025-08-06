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
    """ Initialize a random gaussian PyTree with mimicking the structure of PyTree
    Args:
        rng (PRNGKey): JAX random number generator key
        P (PyTree): PyTree structure to match for the random vector
        rng_gen (function, optional): Random number generator function to use. 
            Defaults to jax.random.normal.
    Returns:
        PyTree: Normalized random vector matching the structure of input P
    Example:
        >>> key = jax.random.PRNGKey(0)
        >>> P = {'a': jnp.zeros(3), 'b': jnp.zeros(2)}
        >>> v0 = InitPowerMethodVector(key, P)
    """
    def generate_entry(rng, p): 
        return rng_gen(rng, p.shape)
    
    rngs = split_rng_for_leaves(rng, P)
    return normalize(jtu.tree_map(generate_entry, rngs, P) )


def initializeHVP(par): 
    """
    Initialize a Hessian-Vector Product (HVP) with zeros.

    This function creates a tree structure with the same shape as the input parameters,
    but filled with zeros.

    Args:
        par (PyTree): A PyTree of parameters whose structure will be mimicked.

    Returns:
        PyTree: A new PyTree with the same structure as `par` but filled with zeros.
    """
    return jtu.tree_map(lambda x : jnp.zeros(x.shape), par)

def standard_parser(batch): 
     return (batch[0], batch[1]) 

def ComputeHVP (loss, par, batch, V, batch_parser=standard_parser):
    """
    Computes the Hessian-vector product (HVP) using forward-mode automatic differentiation.

    This function calculates the product of the Hessian matrix of the loss function
    with respect to the parameters and a given vector V using the Jacobian-vector product (JVP).

    Args:
        loss (callable): Loss function with signature loss(par, x, y) -> float
        par (array-like): Current model parameters
        batch (Any): single batch 
        V (PyTree): Vector(PyTree) to compute product with Hessian
        batch_parser callable: Function to parse batch into inputs and targets. 
                                         Defaults to standard_parser.

    Returns:
        PyTree: Hessian-vector product (H*V)
    """
    x, yhat = batch_parser(batch) 
    grad = jax.grad(lambda p : loss(p, x, yhat), argnums=0) 
    HVP = jax.jvp(grad, (par,), (V,) )[1]
    return HVP 


def PowerMethodStep(loss, par, batch_iterator, V, **kwargs):
    """
    Performs one step of the power method iteration to find the dominant eigenvector and eigenvalue of the Hessian.

    This function implements a single iteration of the power method, computing the product of the Hessian
    with the current vector V using mini-batch processing. It supports momentum for stabilizing the iteration.

    Args:
        loss (Callable): Loss function to compute the Hessian-vector product, with signature loss(par, x, y) -> float
        par (PyTree): Parameters of the model.
        batch_iterator (Iterator): Iterator over batches of data.
        V (PyTree): Current estimate of the eigenvector.
        **kwargs: Additional keyword arguments:
            - show_pbar (bool): Whether to show progress bar. Defaults to False.
            - batch_parser (Callable): Function to parse batch data. Defaults to standard_parser = lambda batch : ( batch[0], batch[1])
            - momentum (float): Momentum coefficient for updating V. Defaults to 0.

    Returns:
        tuple:
            - V_new (PyTree): Updated estimate of the eigenvector (normalized).
            - alpha (float): Current estimate of the eigenvalue.
    """
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
    """
    Performs power method iteration to find the dominant eigenvector of the Hessian matrix.
    Args:
        loss (Callable): Loss function that takes parameters and batch data as input
        par (PyTree): Parameters of the model as a PyTree
        batch_iterable (Iterable): Iterator over batches of training data
        num_iterations (int): Number of power method iterations to perform
        V0 (PyTree, optional): Initial guess for the eigenvector. Must have same structure as par.
            If None, initialized randomly. Defaults to None.
        **kwargs: Additional keyword arguments passed to PowerMethodStep
    Returns:
        tuple: A tuple containing:
            - V (PyTree): Estimated dominant eigenvector 
            - History (dict): Dictionary containing iteration history with keys:
                - 'alpha': List of scaling factors at each iteration, which is an estimate for the square of the dominant eigenvalye
                - 'iteration': List of iteration numbers
                - 'q': List of cosine similarities between successive iterations
    Raises:
        ValueError: If V0 and par don't have the same tree structure
        ValueError: If the leaves of V0 and par don't have the same shapes
    Note:
        The power method iteratively estimates the dominant eigenvector of the Hessian
        matrix without explicitly forming the full matrix. 
    """
    
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

    return V, History

