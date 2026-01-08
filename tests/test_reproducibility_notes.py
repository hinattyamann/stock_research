"""This test documents (does not enforce) reproducibility constraints.

We avoid asserting bitwise-identical deep-learning results because TensorFlow's kernels
can be non-deterministic depending on environment. Instead, we assert that seeding
is callable and doesn't error.
"""

from stock_pred.common_shared import set_seeds

def test_set_seeds_callable():
    set_seeds(42)
