import random
import numpy as np


def set_random_seed(seed: int = 42) -> None:
    """Set random seeds for Python and NumPy.

    Args:
        seed: Seed value used for random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
