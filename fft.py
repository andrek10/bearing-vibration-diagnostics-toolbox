"""FFT
"""
import numpy as np
import pyfftw
from psutil import cpu_count
from scipy.signal import detrend as scipy_detrend

nthreads = cpu_count(logical=False)
pyfftw.interfaces.cache.enable()

def fft(y, Fs, detrend='constant', hann=True, cons=True, debug=False):  
    """
    The scaled amplitude frequency spectrum
    
    Parameters
    ----------
    y : float 1D array
        Signal you want FFT of.
    Fs : float
        Sampling frequency
    Detrend : string, optional
        Detrends the signal using scipy.signal.detrend
        - 'constant' to remove mean value
        - 'linear' to remove least squares fit
        - 'none' to do nothing
    hann : bool, optional
        Add a hanning window if true.
    cons : bool, optional
        Whether conservative part of the spectrum should be returned:

        - True returns Fs/2.56 
        - False returns Fs/2.0
       
    Returns
    -------
    Y : float 1D array
        FFT amplitude
    df : float
        Delta frequency
    """
    
    # Copy input array
    y = np.array(y)

    # Set variables
    n = y.size
    T = n/Fs

    # Check if conservative output is desired
    if cons:
        Fmax = Fs/2.56
    else:
        Fmax = Fs/2.0

    # Get number of lines
    LOR = int(T*Fmax)

    # Remove mean if desired
    if detrend != 'none':
        y = scipy_detrend(y, type=detrend)

    # Apply hanning window
    if hann is True:
        y = np.hanning(y.size)*y

    # Perform DFT
    Y = rawfft(y)
    df = 1.0/T
    return np.abs(Y[0:LOR])*2.0/n, df
    
def rawfft(y):
    """
    Raw FFT of the signal.
    
    Parameters
    ----------
    y : float 1D array
        Signal to get spectrum of
    Fs : float
        Sampling frequency in Hz
    
    Returns
    -------
    Y : float 1D array
        Spectrum values
    df : float
        Delta frequency in Hz
    """
    
    y = np.array(y, copy=True)
    Y_obj = pyfftw.builders.fft(y, auto_align_input=True, auto_contiguous=True, planner_effort='FFTW_ESTIMATE', threads=nthreads, overwrite_input=True)
    return Y_obj()
    
def rawifft(Y):
    """
    Raw inverse FFT

    Parameters
    ----------
    Y : float 1D array
        Spectrum to inverse

    Returns
    -------
    y : float 1D array
        Time domain signal
    """
    
    Y_obj = pyfftw.builders.ifft(Y, planner_effort='FFTW_ESTIMATE')
    return Y_obj()
