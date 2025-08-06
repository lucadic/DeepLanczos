import jax.tree_util as jtu
import jax.random as jar


def initialize_random_entry(p, rng, M=1, rnd_gen=jar.normal):
    shape = p.shape
    total_shape = (M, *shape)
    return rnd_gen(rng, total_shape)


def compute_shapes(par): 
    return jtu.tree_map(lambda x: x.shape, par)


def split_rng_for_leaves(rng, tree):
    leaves, treedef = jtu.tree_flatten(tree)
    num_leaves = len(leaves)
    new_keys = jar.split(rng, num_leaves)
    return jtu.tree_unflatten(treedef, new_keys)


def GenerateRandomVectors(par, rng, M, rnd_gen=jar.normal):
    rngs = split_rng_for_leaves(rng, par)

    def generate(p, r):
        return initialize_random_entry(p, r, M, rnd_gen=rnd_gen)

    return jtu.tree_map(generate, par, rngs)

def accumulateHVP(HVP_accumulated, HVP): 
    return jtu.tree_map(lambda x, y: x+y, HVP_accumulated, HVP)

def check_shapes(X, Y): 
    shapes_check = jtu.tree_map(lambda x, y : x.shape == y.shape, X, Y) 
    return jtu.tree_reduce(lambda s1, s2 : s1 and s2, shapes_check, initializer=True)
    