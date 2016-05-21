import numpy as np

def spectral_radius(A):
    """
    Return the spectral radius of matrix `A`.

    """
    return np.max(abs(np.linalg.eigvals(A)))
