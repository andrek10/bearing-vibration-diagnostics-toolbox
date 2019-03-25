'''
Statistical functions
'''

from copy import deepcopy
from math import log

import matplotlib.pyplot as plt
import numpy as np
from numba import jit, njit

from .misc import pretty
from .signal import envelope, fftwconvolve


def arresidual(t, y, a):
    '''
    Returns the residual of the autoregressive model with coefficients a
    
    Parameters
    ----------
    t : float 1D array
        Time signal
    y : float 1D array 
        Signal to filter
    a : float 1D array 
        AR model coeffs
        
    Returns
    -------
    t : float 1D array 
        New time signal
    y : float 1D array 
        Filtered signal
    '''

    y = fftwconvolve(y, a, 'valid')
    N = a.size - 1
    t = t[N:]
    return t, y

def arresponse(t, y, a):
    '''
    Returns the predicted response of the autoregressive model with coeffs a
    
    Parameters
    ----------
    t : float 1D array
        Time signal
    y : float 1D array
        Signal
    a : float 1D array
        AR model coeffs
        
    Returns
    -------
    t : float 1D array
        New time signal
    y : float 1D array 
        Filtered signal
    '''

    ynoise = fftwconvolve(y, a, 'valid')
    N = a.size - 1
    y = y[N:] - ynoise
    t = t[N:]
    return t, y

def armodel(y, p, Crit=0, debug=False):
    '''
    This function tries to remove stationary signals by estimating an
    autoregressive model on the vibration signal. Afterwards this estiamte can
    be subtracted from the original signal using arresidual()

    Parameters
    ----------
    y : float 1D array
        Vibration data.
    p : int 
        Maximum number of filter coefficients
    Crit : int, optional
        Criterion for choosing optimal p:

        - 0 uses Akaike Information Criterium (AICc)
        - 1 uses Bayesian Information Criterium (BIC)
    
    debug : boolean, optional
        Choose if debug information should be returned

    Returns
    -------
    aopt : float 1D array
        Optimal AR model parameters
    popt : int
        Optimal model order

    See also
    --------
    arresidual() - Return the residual (random) signal
    arresponse() - Returns the autoregressive model response
    '''

    y = y - y.mean()
    RR = fftwconvolve(np.flipud(y), y)
    RR_start = RR.size // 2

    # Actual AR model
    abest, sigma, AIC, apmax, popt = _autoRegressiveFilter_v4(RR[RR_start:RR_start+p+1], p, Crit, y.size)
    aopt = np.concatenate((np.ones(1), abest[0:popt]))
    apmax = np.concatenate((np.ones(1), apmax))
        
    if debug is True:
        return aopt, popt, apmax, AIC, sigma
    else:
        return aopt, popt  
    
def EHNR(x, Fs = 1.0, debug=False):
    '''
    Get Envelope Harmonic-to-noise ratio 
    Based on:
    Xu, X., Zhao, M., Lin, J., & Lei, Y. (2016). 
    Envelope harmonic-to-noise ratio for periodic impulses 
        detection and its application to bearing diagnosis. 
    Measurement, 91, 385-397.

    Parameters
    ----------
    x : float 1D array
        Signal
    Fs : float, optional
        Sampling frequency
    debug : boolean, optional
        Whether debug information is returned

    Returns
    -------
    EHNR : float
        The EHNR value
    '''

    if x.size % 2 == 1:
        x = x[0:-1]
    Env_prime = envelope(x)
    Env = Env_prime - np.mean(Env_prime)
    
    t_i = x.size/2 - 1
    dt = 1.0/Fs
    temp = Env[:t_i]
    r_Env = fftwconvolve(Env[:2*t_i], temp[::-1], 'valid')*dt
    i_rem = int(t_i/20.0)
    for i in range(i_rem, t_i):
        der = r_Env[i] - r_Env[i-1]
        if der > 0.0:
            i_rem = i
            break
    if i == t_i - 1:
        return 0.0
    tauMax_i = np.argmax(r_Env[i_rem:]) + i_rem
    r_EnvMax = r_Env[tauMax_i]
    if debug is True:
        plt.figure()
        axes = plt.gca()
        plt.plot(r_Env)
        plt.title('i_rem = %i' % (i_rem))
        ylim = axes.get_ylim()
        plt.plot([i_rem, i_rem], [ylim[0], ylim[1]], '-g')
        plt.plot([tauMax_i, tauMax_i], [ylim[0], ylim[1]], '--r')
        plt.show()
    EHNR = r_EnvMax/(r_Env[0] - r_EnvMax)
    return EHNR

@jit(nopython=True, cache=True)
def _autoRegressiveFilter_v4(RR, pmax, Crit, signalSize):
    '''
    Solves the Yule-Walker equations for autoregressie model
    '''
    # Make necessary arrays for calculation
    abest = np.zeros(pmax)
    aold = np.zeros(pmax)
    anew = np.zeros(pmax)
    sigma = np.zeros(pmax)
    AIC = np.zeros(pmax)
    AIC_min = 1e10
    popt = 0

    # Initialize AR solver
    anew[0] = -RR[1]/RR[0]
    sigma[0] = (1.0 - anew[0]**2)*RR[0]

    # Initialzie best AR model
    AIC[0] = signalSize*(log(sigma[0] / signalSize) + 1.0) + 2.0*(0 + 1 + 1)*signalSize/(signalSize - (0+1) - 2)
    AIC_min = AIC[0]
    abest[0] = anew[0]
    popt = 1

    # Recurrsively iterate through
    for k in range(1, pmax):
        for j in range(0, k):
            aold[j] = anew[j]
        temp = 0.0
        for j in range(0, k):
            temp += aold[j]*RR[k-j]
        anew[k] = -(RR[k+1] + temp)/sigma[k-1]
        for i in range(0, k):
            anew[i] = aold[i] + anew[k]*aold[k-i-1]
        sigma[k] = (1 - anew[k]*anew[k])*sigma[k-1]
        AIC[k] = signalSize*(log(sigma[k] / signalSize) + 1) + 2*(k+1 + 1)*signalSize/(signalSize - k+1-2)
        if AIC[k] < AIC_min:
            AIC_min = AIC[k]
            popt = k + 1
            for j in range(0, k+1):
                abest[j] = anew[j]
    
    # Return
    return abest, sigma, AIC, anew, popt

def _checkOccurences(rho, tol, printOccurences, skipSignals):
    '''
    Checks occurences of co-variances being over a thrshold
    '''

    occurences = np.zeros(n)
    for i in range(0, n):
        for j in range(i+1, n):
            if np.abs(rho[i, j]) >= tol:
                occurences[i] += 1
                occurences[j] += 1
                if printOccurences is True:
                    print('W: Linear dependency between signals  %i-%i' % (i, j))
    if printOccurences is True:
        print('Occurences:')
        pretty(occurences[:,None])
    
    return occurences

def covariance(A, printSingular=False, tol=0.9, skipSignals=[]):
    '''
    Compute the covariance of columns in matrix A

    Parameters
    ----------
    A : array
        [m,n] array with m observatios and n signals.
    printSingular : bool, optional
        Print list of singular signals

    Returns
    -------
    rho : array
        Covariance matrix
    occurences : array
        How many other signals each signal is 
        similar to.
    '''

    n = A.shape[1]
    rho = np.zeros((n, n))
    for i in range(0, n):
        if i in skipSignals: next
        for j in range(0, n):
            if j in skipSignals: next
            temp = pearsonr(A[:, i], A[:, j])
            rho[i, j] = temp[0]

    occurences = _checkOccurences(rho, tol, printSingular, skipSignals)

    return rho, occurences

def maximizeUncorrelatedSignals(A, tol=0.9):
    '''
    Maximize number of signals such that all are uncorrelated 
    according to the tolerance.

    A : array, or list of arrays
        [m,n] array with m observatios and n signals.
        If list, n must be equal on all arrays
    tol : float, optional
        Tolerance for covariance
    '''

    # Initialize
    if type(A) is np.ndarray:
        operation = 0
        A = np.array(A)
        ABest = np.array(A)
        N = A.shape[1]
    elif type(A) is list:
        operation = 1
        A = deepcopy(A)
        ABest = deepcopy(A)
        N = A[0].shape[1]
    else:
        print('Wrong input A')
        return None
    nMax = 0

    skipSignals = []
    for j in range(N, 1):
        if operation == 0:  
            rho, occurences = covariance(A, printSingular=False, tol=tol, skipSignals=skipSignals) 
            n = np.sum(occurences == 0.0)
            if n > nMax:
                nMax = n
                ABest = np.array(A)
            I = np.ones(j, bool)
            temp = np.argsort(occurences)

            I[temp[-1]] = False
            
            
        elif operation == 1:
            rho = np.empty((j, j))
            for k in range(0, len(A)):
                rhoTemp, occurencesTemp = covariance(A, printSingular=False, tol=tol)
                rho += rhoTemp
            rho /= len(A)
            occurences = _checkOccurences(rho, tol, printOccurences=False)        
            n = np.sum(occurences == 0.0)
            if n > nMax:
                nMax = n
                ABest = deepcopy(A)

@njit(cache=True)
def _spearmanWorker(temp1, temp2):
    '''
    Calculate the spearman coefficient
    '''
    return np.sum((temp1)*(temp2)) / np.sqrt(np.sum((temp1)**2)*np.sum((temp2)**2))

def spearman(x1):
    '''
    Computes the spearman coefficient of input x1 np.array
    Assumes the comparison vector is linearly increasing. 

    Parameters
    ----------
    x1 : float 1D array
        The signal to calculate Spearman coefficient of
    
    Returns
    -------
    spearman : float
        Spearman coefficient
    '''

    x1rankMean = float(x1.size - 1)/2.0
    temp1 = np.argsort(x1)[::-1] - x1rankMean
    temp2 = np.arange(x1.size-1, -1, -1) - x1rankMean
    return _spearmanWorker(temp1, temp2)


@njit(cache=True)
def _percentileWorker(x, y, yp):
    '''
    Calculates the percentile
    '''
    if yp <= y[0]:
        return x[0]
    x1 = x[0]
    y1 = y[0]
    for i in range(1, x.size):
        x2 = x[i]
        y2 = y[i]
        if y1 <= yp <= y2:
            xp = (yp - y1)*(x2 - x1)/(y2 - y1) + x1
            break
        y1 = y2
        x1 = x2
    return xp
    
def percentile(v, p, w=None):
    '''
    Gets the p percentile of a PDF with weights w and values v
    0.0 <= p <= w.sum
    if w is None, w.sum == 1.0

    Parameters
    ----------
    v : float 1D array
        Value samples
    p : float
        Percentile
    w : float 1D array, optional
        Weights of the value samples

    Returns
    -------
    percentile : float
        The percentile
    '''
    assert 0.0 <= p <= 1.0
    if w is None:
        w = np.ones(v.size)/v.size
    else:
        assert w.size == v.size
    I = np.argsort(v)
    return _percentileWorker(v[I], np.cumsum(w[I]), p)
