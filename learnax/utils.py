import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_map, tree_flatten

def tree_stack(trees):
    def _stack(*v):
        try:
            if type(v[0]) == np.ndarray or type(v[0]) == jnp.ndarray:
                return np.stack(v)
            else:
                return None
        except:
            breakpoint()
    return tree_map(_stack, *trees)


def tree_unstack(tree):
    leaves, treedef = tree_flatten(tree)
    return [treedef.unflatten(leaf) for leaf in zip(*leaves, strict=True)]
