"""
misc
"""

import numpy as np
import pandas as pd
from numba import njit


def primefactors(number):
    """
    Determine all prime factors for the input number

    Parameters
    ----------
    number : int
        Number to get prime factors of

    Returns
    -------
    factors : 1D array of int
        Prime factors
    """

    number = int(number)
    i = 2
    factors = []
    while i * i <= number:
        if number % i:
            i += 1
        else:
            number //= i
            factors.append(i)
    if number > 1:
        factors.append(number)
    return np.array(factors, dtype=int)

@njit(cache=True)
def _XOR(x):
    XOR = np.zeros(x.size, dtype=int)
    for i in range(1, x.size):
        if not x[i] == x[i-1]:
            XOR[i] = 1
    return XOR

def pretty(A, precision=2, maxcol=1000, maxrow=1000):
    pd.options.display.max_columns = maxcol
    pd.options.display.max_rows = maxrow
    pd.options.display.precision = precision
    print(pd.DataFrame(A))
