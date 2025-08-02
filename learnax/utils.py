import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_map_with_path, tree_flatten
import jaxlib


def tree_stack(trees):
    def _stack(k, *v):
        try:
            if (
                type(v[0]) == np.ndarray
                or type(v[0]) == jnp.ndarray
                or type(v[0]) == jax.Array
                or type(v[0]) == list
                or type(v[0]) == jaxlib.xla_extension.ArrayImpl
            ):
                return np.stack(v)
            else:

                return None
        except:
            print(f"Error stacking at key: {k}, values: {v}")
            breakpoint()

    return tree_map_with_path(_stack, *trees)


def tree_unstack(tree):
    leaves, treedef = tree_flatten(tree)
    return [treedef.unflatten(leaf) for leaf in zip(*leaves, strict=True)]
