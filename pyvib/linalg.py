"""
Linear algebra
"""

import numpy as np
from numba import njit
from scipy.linalg import hankel
from sklearn.utils.extmath import randomized_svd


def SVD(x, m):
    '''
    Produce the singular value decomposition of a signal
    Uses the scikit learn.randomized SVD function

    Parameters
    ----------
    x : array 1D float
        The signal to decompose
    m : int
        Number of singular values wanted

    Returns
    -------
    U : array 2D float
    sigma : array 1D float
        The singular values ranging from highest to lowest energy
    V : array 2D float

    See also
    --------
    get_SVDxi - Use get to recombine a singular value back to time-domain
    '''

    A = hankel(x[0:m], x[m-1:])
    U, sigma, V = randomized_svd(A, m)
    return U, sigma, V
    
def get_SVDxi(U, sigma, V, i):
    '''
    Estimate the i'th singular value composition using the diagonal mean sum

    Parameters
    ----------
    U : array 2D float
    sigma : array 1D float
        The singular values
    V : array 2D float
    i : int
        The i'th singular value to recompose. 0 < i < Sigma.size
    '''

    Ai = sigma[i]*(U[:,i][:,None].dot(V[i,:][None,:]))
    m = Ai.shape[0]
    N = Ai.shape[1]
    xi = np.zeros(m+N-1)
    _get_SVDxiJit(Ai, xi, m, N)
    return xi

@njit(cache=True)
def _get_SVDxiJit(Ai, xi, m, N):
    '''
    JIT worker for get_SVDxi
    '''
    for i in range(0, xi.size):
        kstart = N - xi.size + i - 1
        elems = 0
        for j in range(m - 1, -1, -1):
            k = kstart + m - j
            if k > -1 and k < N:
                xi[i] += Ai[j, k]
                elems += 1
        xi[i] /= elems
