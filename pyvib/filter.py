"""
Filer design
"""

import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import decimate as scipydecimate
from scipy.signal import firwin

from .fft import rawfft, rawifft
from .misc import primefactors
from .plt import figure
from .signal import fftwconvolve


def get_filterbankfilters(N, fc=0.25):
    '''
    Make filters for filterbank decomposition and recomposition
    These are even order FIR filters

    Parameters
    ----------
    N : int
        The filter length. Must be even number
    fc : float
        Normalized cutoff frequency <0.0, 0.5>

    Returns
    -------
    f0, f1, h0, h1 : arrays of float
        The filter kernels
    '''
    
    # N must to even to make the best filter!
    assert N % 2 == 0

    def spectral_flip(h):
        g = np.copy(h)
        g[0::2] = -g[0::2]
        return g

    h0 = firwin(N+1, fc, nyq=0.5)
    h1 = spectral_flip(h0)
    f0 = spectral_flip(h1)*2
    f1 = - spectral_flip(h0)*2
    return h0, h1, f0, f1

def get_filterbankfilters_kurtogram(N=16):
    '''
    Acquire the filterbank filters used in:
    Antoni, Jerome. "Fast computation of the kurtogram for the detection of transient faults." 
    Mechanical Systems and Signal Processing 21.1 (2007): 108-124.

    Parameters
    ----------
    N : int
        Number of filterbank coefficients
    
    Returns
    -------
    h : float 1D array
        Lowpass filter
    g : float 1D array
        Highpass filter
    '''

    fc = 0.4
    h = firwin(N+1,fc)*np.exp(2*1j*np.pi*np.arange(0, N+1)*0.125)
    n = np.arange(2, N+2, 1)
    g = np.flipud(h)[1:]*(-1.0)**(1-n)
    return h, g

def _fbdecompose(x, h0, h1):
    x0 = fftwconvolve(x, h0, 'same')
    x1 = fftwconvolve(x, h1, 'same')
    v0 = np.copy(x0[0::2])
    v1 = np.copy(x1[1::2])
    xsize = x0.size
    return v0, v1, xsize

def _fbcompose(x0, x1, f0, f1, xsize):
    c0 = np.zeros(xsize)
    c0[0::2] = np.copy(x0)
    c1 = np.zeros(xsize)
    c1[1::2] = np.copy(x1)
    y0 = fftwconvolve(c0, f0, 'same')
    y1 = fftwconvolve(c1, f1, 'same')
    return y0+y1

def filterbank_decompose(x, h0, h1, level):
    '''
    Decompose a signal using supplied filters for a certain numebr of levels

    Parameters
    ----------
    x : float 1D array
        Signal
    h0, h1 : float 1D arrays
        Filter kernels
        h0 is low-pass, h1 is highpass
    level : int
        The filter decomposition level

    Returns
    -------
    xbank : list of float 1D arrays
        The filter-bank coefficients ranging from lowest frequency to highest frequency
    xsizes : list of lists of integers
        The sizes of signals before decomposing. 
        Only needed for recomposing using filterbank_compose()

    See also
    --------
    get_filterbankfilters() : Makes the h0 and h1 filter kernel
    filterbank_compose() : Re-combposes an xbank into a signal
    '''
    
    xbank = [x,]
    xsizes = []
    for i in range(0, level):
        xnew = []
        xsizes.append([])
        for j in range(0, len(xbank)):
            v0, v1, xsize = _fbdecompose(xbank[j], h0, h1)
            xnew.append(v0)
            xnew.append(v1)
            xsizes[i].append(xsize)
        xbank = xnew
    return xbank, xsizes

def filterbank_compose(xbank, f0, f1, xsizes):
    '''
    Recompose the filter bank to a single signal

    Parameters
    ----------
    xbank : float 1D array
        The filterbank
    f0, f1 : float 1D arrays
        The filter kernels
    xsizes : list of list of ints
        The sizes of signals before decomposing

    Returns
    -------
    x_hat : float array
        The recomposed signal. Should be close to the original
        signal x after applying the lag
    lag : int
        The lag of the recomposed signal
        Should ideally use x_hat[lag:-lag] after recomposition
        x_hat[lag:-lag] approximates x[0:-lag*2]
    '''
    
    level = int(np.log2(len(xbank)))
    for i in range(0, level):
        xbank_new = []
        for j in range(0, len(xbank), 2):
            xsize = xsizes[len(xsizes)-i-1][j//2]
            y = _fbcompose(xbank[j], xbank[j+1], f0, f1, xsize)
            xbank_new.append(y)
        xbank = xbank_new
    lag = int(2**level - 1)
    return xbank[0], lag
    
def waveletfilter(f0, sigma, Fs, N):
    '''
    Constructs the frequency transformed wavelet filter. Can be used to
    filter a frequency transformed signal by taking Y*Ksi.

    Parameters
    ----------
    f0 : float
        The center frequency for the bandpass filter in Hz
    sigma : float
        The width of the filter in Hz
    Fs : float
        The sampling frequency of the signal in Hz
    N : int
        The number of samples in the signal in Hz

    Returns
    -------
    Ksi : float 1D array
        Filter in the frequency domain.
    '''
    
    dt = 1.0/Fs
    T = dt*float(N)
    df = 1.0/T
    f = np.arange(0, Fs/2.0, df)
    Ksi = np.exp(-(np.pi**2.0/sigma**2.0)*(f - f0)**2.0)
    Ksi = np.concatenate((Ksi, np.zeros(N - Ksi.size)))
    return Ksi

def blinddeconvolution(z, L, part=1.0, k=4.0, maxIter=1000, maxMu=2.0, stopCrit=0.01, debug=False):
    '''
    Iteratively identifies a filter g that deconvolves the filter h 
    originally applied to z to return the deconvolved signal x.
    The iterator tries to maximize the kurtosis (impulsivity) of the
    deconvolved signal.
    The deconvolution is afterwards performed using:
    x = pyvib.signal.fftwconvolve(z, gNew, 'valid')

    Parameters
    ----------
    z : float 1D array
        Signal to deconvolve
    L : int
        Length of filter
    part : float, optional
        Percentage of the data to train the filter on.
        Must be within <0, 1>
    k - float, optional
        Exponent of the objective. 4 gives kurtosis
    maxIter : int, optional
        Maximum number of iterations to run
    maxMu : float, optional
        Maximum training coefficient
    stopCrit : float, optional
        Stopping criterion
    debug : boolean, optional
        Print progression if true

    Returns
    -------
    gNew : float 1D array
        Filter kernel that deconvolves the signal
    '''
    
    temp = np.ones(L)
    temp[::2] *= -1.0
    gNew = temp*np.random.rand(L)
    g = np.copy(gNew)
    assert 0.0 < part <= 1.0
    i1 = 0
    i2 = i1 + int(part*z.size)
    N = i2 - i1
    Rxx = fftwconvolve(np.flipud(z[i1:i2]), z[i1:i2])
    cc = Rxx[N-1:N+L-1]
    mu = 1.0
    eOld = 1e15
    for i in range(0, maxIter):
        g = g + mu*(gNew - g)
        y = fftwconvolve(z[i1:i2], g, 'valid')
        alpha = np.sum(y**2.0)/np.sum(y**k)
        if y.size > (i2-i1):
            temp = np.flipud(fftwconvolve(np.flipud(y**(k-1)), z[i1:i2], 'valid'))
        else:
            temp = np.flipud(fftwconvolve(np.flipud(z[i1:i2]), y**(k-1), 'valid'))
        b = alpha*temp
        gNew = solve_toeplitz(cc, b)
        e = np.sqrt(np.sum((g - gNew)**2.0))
        if e > eOld:
            mu = 1.0
        else:
            mu *= 1.1
        if mu >= maxMu:
            mu = maxMu
        eOld = e
        if debug is True:
            print('i = %i, e = %.3f, mu = %.3f' % (i, e, mu))
        if e < stopCrit:
            break

    return gNew

def filterdesign(Yh, M, plot=True):
    '''
    Design a FIR filter that matches a frequency response

    Parameters
    ----------
    Yh : float 1D array
        The amplitude specrum to match
    M : int
        Number of coefficients to use in the filter
    plot : boolean, optional
        Whether the resulting filter should be plotted

    Resturns
    --------
    h : float 1D array
        The designed FIR filter kernel
    '''

    if M % 2 != 0:
        M += 1
    if M > Yh.size:
        M = Yh.size - 2
        if M % 2 != 0:
            M -= 1

    y = rawifft(Yh)
    f = np.fft.fftfreq(y.size)
    
    h = np.zeros(M+1, dtype='complex')
    h[0:M//2] = y[-M//2:]
    h[M//2:M+1] = y[0:M+1-M//2]
    Hamm = np.hamming(M+1)
    h[0:M+1] *= Hamm

    if plot is True:
        hplot = np.zeros(y.size, dtype='complex')
        hplot[0:h.size] = h
        Yh2 = rawfft(hplot)
        fig, ax = figure(1, 1, width=500.0, height=300.0)
        ax.plot(f, Yh/Yh.max(), 'b', label='Input')
        ax.plot(f, np.abs(Yh2)/Yh2.max(), '--r', label='Returned')

    return h

def decimate(y, decimatelist):
    '''
    Apply decimation of a signal y with time t by applying an IIR filter
    with the decimation factor given by all items in decimatelist

    Parameters
    ----------
    y : float 1D array
        The signal
    decimatelist : int, array_like
        Acquire this list using get_decimatelist()

    Returns
    -------
    yd : float 1D array
        Decimated signal

    See Also
    --------
    get_decimatelist()
    '''

    decimatelist = np.asarray(decimatelist, dtype=int)
    for i in range(0, decimatelist.size):
        y = scipydecimate(y, int(decimatelist[i]), zero_phase=False)
    return y

def get_decimatelist(desireddecimation, maxdec=12, step=-1):
    '''
    Generate a decimation list for using the decimate function

    Parameters
    ----------
    desireddecimation : int
        Desired decimation factor in total
        Going from a sample frequency of 50 kHz to 10 kHz is a factor of 5
    maxdec : int, optional
        Maximum decimation per iteration
        Defaults to 12
    direction : int, optional
        Step to make if a decimation factor is not suitable

    Returns
    -------
    decimatelist : list
        Decimation factors to follow per decimation iteration.

    See Also
    --------
    decimate
    '''

    desireddecimation = int(desireddecimation)
    if desireddecimation <= maxdec:
        return np.array([desireddecimation,], dtype=int)
    dec = desireddecimation
    while True:
        decL = primefactors(dec)
        if np.greater(decL, maxdec).any():
            dec += step
        else:
            break
        if dec <= maxdec:
            decL = np.array([dec,], dtype=int)
    return decL

def cpw(x):
    '''
	Removes synchronous parts of a signal using Cepstrum Pre-Whitening

	Parameters
	----------
    x : float 1D array
        Signal

	Returns
	-------
    xPW : float 1D array
        Whitened signal
    '''

    spec = rawfft(x - np.mean(x))
    spec[np.less(spec, 1e-15)] = 1e-15
    temp = spec/np.abs(spec)
    if np.sum(np.isnan(temp)) > 0:
        print('CPW Warning: NaN found in FFT of x!')
        print(spec[np.isnan(temp)])
        temp[np.isnan(temp)] = 0.0
    temp2 = rawifft(temp)
    xPW = np.real(temp2)
    return xPW


class IIRFilter():
    '''
    An IIR filter object that can update per sample.
	
    Parameters
    ----------
    b : float 1D array
        Filter coefficients
    a : float 1D array
        Filter coefficients
    
    Object functions
    ----------------
    update() : Filter the next sample

    See also
    --------
    scipy.signal.butter() - Create an IIR filter
    '''
	
    def __init__(self, b, a):
        self._M = a.size
        self._N = b.size
        self._a = a
        self._b = b
        self._x = np.zeros(self._M)
        self._y = np.zeros(self._M)
        self._iters = 0
    def update(self, xin):
        '''
        Updates the filter with a new sample
        
        Parameters
        ----------
        xin : float
            The new sample
        
        Returns
        -------
        xout : float
            The filtered sample
        '''
        
        if self._iters < self._M:
            # Initialize
            i = self._iters
            self._x[i] = xin
            for j in range(0, i+1, 1):
                self._y[i] += self._x[i-j]*self._b[j]
            for j in range(1, i+1, 1):
                self._y[i] -= self._y[i-j]*self._a[j]
            self._y[i] /= self._a[0]
        else:
            # Update
            self._y[0:self._M-1] = self._y[1:self._M]
            self._y[-1] = 0.0
            self._x[0:self._M-1] = self._x[1:self._M]
            self._x[-1] = xin
            i = self._M - 1
            for j in range(0, self._N):
                self._y[-1] += self._x[i-j]*self._b[j]
            for j in range(1, self._M):
                self._y[-1] -= self._y[i-j]*self._a[j]
            self._y[-1] /= self._a[0]

        self._iters += 1
        return self._y[i]
