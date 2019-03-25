"""
acoustics
"""
import numpy as np
import pyfftw

from .fft import rawfft, rawifft

pyfftw.interfaces.cache.enable()

def cepstrum(x):
    """
	Get cepstrum of a signal
	
	Parameters
	----------
    x : float 1D array
        Signal
	
	Returns
	-------
    xc : float 1D array
        Cepstrum of the signal
	"""
    return np.abs( rawifft( np.log( np.abs( rawfft(x) )**2.0 ) ) )**2.0
