"""
Different features that can be calculated with the vibration signal
"""

from math import log

import matplotlib.pyplot as plt
import numpy as np
from numba import jit, njit, prange
from scipy.stats import pearsonr

from .signal import envelope, fftwconvolve


def rms(y):
    """
    Get RMS vlaue

    Parameters
    ----------
    y : float 1D array
        Signal

    Returns
    -------
    RMS : float
        RMS
    """

    rms = np.sqrt(np.sum(y**2)/y.size)
    return rms

def kurtosis(x):
    """
    Get kurtosis value

    Parameters
    ----------
    x : float 1D array
        Signal

    Returns
    -------
    K : float
        Kurtosis
    """

    x2 = np.abs(x - np.mean(x))**2.0
    E = np.mean(x2)
    K = np.mean(x2**2.0)/E**2.0
    return K

def standardmoment(x, k):
    """
    Get standard moment of choice

    Parameters
    ----------
    x : float 1D array
        Signal
    k : int
        Desired moment

    Returns
    SM : float
        Standard moment of choice
    """

    xk = (x - np.mean(x))**k
    x2 = (x - np.mean(x))**2
    SM = np.mean(xk)/np.mean(x2)**(float(k)/2.0)
    return SM

def absoluteMean(x):
    """
    Get absolute mean of signal

    Parameters
    ----------
    x : float 1D array
        Signal
    """

    return np.mean(np.abs(x))

def peakToPeak(x):
    """
    Peak-to-peak of signal

    Parameters
    ----------
    x : float 1D array
        Signal
    """

    return np.max(x) - np.min(x)

def squareMeanRoot(x):
    """
    Square mean root of signal

    Parameters
    ----------
    x : float 1D array
        Signal
    """

    return np.sum(np.sqrt(np.abs(x)))**2

def waveformLength(x):
    """
    Waveform length of signal

    Parameters
    ----------
    x : float 1D array
        Signal

    Taken from paper:
        Nayana, B. R., and P. Geethanjali.
        "Analysis of Statistical Time-Domain Features Effectiveness
        in Identification of Bearing Faults From Vibration Signal."
        IEEE Sensors Journal 17.17 (2017): 5618-5625.
    """

    return np.sum(np.abs(np.diff(x)))

def willsonAmplitude(x, epsilon=0.5):
    """
    Willson amplitude

    Parameters
    ----------
    x : float 1D array
        Signal

    Taken from paper:
        Nayana, B. R., and P. Geethanjali.
        "Analysis of Statistical Time-Domain Features Effectiveness
        in Identification of Bearing Faults From Vibration Signal."
        IEEE Sensors Journal 17.17 (2017): 5618-5625.
    """

    return np.sum(np.diff(np.abs(x)) > epsilon)

def zeroCrossing(x, epsilon=0.5):
    """
    Zero crossing og signal

    Parameters
    ----------
    x : float 1D array
        Signal
    epsilon : float, optional

    Taken from paper:
        Nayana, B. R., and P. Geethanjali.
        "Analysis of Statistical Time-Domain Features Effectiveness
        in Identification of Bearing Faults From Vibration Signal."
        IEEE Sensors Journal 17.17 (2017): 5618-5625.
    """

    temp1 = np.greater(x[:-1], 0)*np.greater(0, x[1:])
    temp2 = np.less(x[:-1], 0)*np.less(0, x[1:])
    temp3 = np.greater(np.abs(np.diff(x)), epsilon)
    return np.sum(np.logical_and(np.logical_or(temp1, temp2), temp3))

def slopeSignChange(x, epsilon=0.5):
    """
    Slope sign change

    Parameters
    ----------
    x : float 1D array
        Signal
    epsilon : float, optional

    Taken from paper:
        Nayana, B. R., and P. Geethanjali.
        "Analysis of Statistical Time-Domain Features Effectiveness
        in Identification of Bearing Faults From Vibration Signal."
        IEEE Sensors Journal 17.17 (2017): 5618-5625.
    """

    temp1 = np.greater(x[1:-1], x[:-2])*np.greater(x[1:-1], x[2:])
    temp2 = np.less(x[1:-1], x[:-2])*np.less(x[1:-1], x[2:])
    temp3 = np.greater(np.abs(np.diff(x[:-1])), epsilon)
    return np.sum(np.logical_and(np.logical_or(temp1, temp2), temp3))

def shapeFactor(x):
    """
    Shape factor

    Parameters
    ----------
    x : float 1D array
        Signal
    """

    return rms(x)/absoluteMean(x)

def crestFactor(x):
    """
    Crest factor

    Parameters
    ----------
    x : float 1D array
        Signal
    """

    return np.max(x)/rms(x)

def impulseFactor(x):
    """
    Impulse factor

    Parameters
    ----------
    x : float 1D array
        Signal
    """

    return np.max(x)/absoluteMean(x)

def clearanceFactor(x):
    """
    Clearance factor

    Parameters
    ----------
    x : float 1D array
        Signal
    """

    return np.max(x)/squareMeanRoot(x)

def skewnessFactor(x):
    """
    Skewness factor

    Parameters
    ----------
    x : float 1D array
        Signal
    """

    return standardmoment(x, 3)/rms(x)**3

def kurtosisFactor(x):
    """
    Kurtosis factor

    Parameters
    ----------
    x : float 1D array
        Signal
    """

    return standardmoment(x, 4)/rms(x)**4

def rootMeanSquareFrequency(f, Y):
    """
    Root mean square frequency

    Parameters
    ----------
    f : float 1D array
        Frequency of FFT
    Y : float 1D array
        Amplitude of FFT
    """

    return ( np.sum(f**2*Y)/np.sum(Y) )**0.5

def frequencyCenter(f, Y):
    """
    Frequency center of spectrum

    Parameters
    ----------
    f : float 1D array
        Frequency of FFT
    Y : float 1D array
        Amplitude of FFT
    """

    return np.sum(f*Y)/np.sum(Y)

@njit(cache=True)
def _maxdist(x_i, x_j):
    ma = 0.0
    temp = 0.0
    for k in range(0, x_i.shape[0]):
        temp = abs(x_i[k] - x_j[k])
        if temp >= ma:
            ma = temp
    return ma
    # return np.max(np.abs(x_i - x_j))

@njit(cache=True)
def _phi(U, m, r):
    N = U.size
    iterRange = N - m + 1
    iterRangeFloat = float(iterRange)
    a = 0.0
    for i in prange(0, iterRange):
        hits = 0.0
        for j in range(0, iterRange):
            if _maxdist(U[i:i+m], U[j:j+m]) <= r:
                hits += 1.0
        a += log(hits/iterRangeFloat)
    return a/iterRangeFloat

def approximateEntropy(U, N=1000, m=2, r=None):
    """
    Approximate the entropy of a signal

    Parameters
    ----------
    U : float 1D array
        Signal
    N : int, optional
    m : int, optional
    r : float, optional

    https://en.wikipedia.org/wiki/Approximate_entropy
    Default values taken from:
    Yan, Ruqiang, and Robert X. Gao.
    "Approximate entropy as a diagnostic tool for machine health monitoring."
    Mechanical Systems and Signal Processing 21.2 (2007): 824-839.
    """

    if r is None:
        r = 0.4*U.std()
    temp1 = _phi(U[:N], m+1, r)
    temp2 = _phi(U[:N], m, r)
    return abs(temp1 - temp2)

def SDofIHC(x):
    """
    Standard deviation of inverse hyperbolic cosine

    Parameters
    ----------
    x : float 1D array
        Signal

    Taken from paper
    A Model-Based Method for Remaining Useful Life Prediction of Machinery
    Yaguo Lei et al.
    """
    return np.std(np.log(x + np.sqrt(x**2 - 1)))

def SDofIHS(x):
    """
    Standard deviation of inverse hyperbolic sine

    Parameters
    ----------
    x : float 1D array
        Signal

    Taken from paper
    A Model-Based Method for Remaining Useful Life Prediction of Machinery
    Yaguo Lei et al.
    """
    return np.std(np.log(x + np.sqrt(x**2 + 1)))

def medianFrequency(Y, df):
    """
    Median frequency of a spectrum

    Parameters
    ----------
    Y : float 1D array
        Spectrum aplitude
    df : float
        Frequency spacing between bins
    """

    cumsum = np.cumsum(Y)
    return np.argmin(np.abs(cumsum - 0.5*cumsum[-1]))*df

def myoPulsePercentage(x, eps=5.0):
    """
    Myo pulse percentage
    Sum of all impulses greater than a threshold eps

    Parameters
    ----------
    x : float 1D array
        Signal
    eps : float, opional
        Threshold
    """

    return np.sum(np.greater(np.abs(x), eps))

def LOG(x):
    """
    Exponential of the mean absolute logarithm

    Parameters
    ----------
    x : float 1D array
        Signal
    """

    return np.exp(np.mean(np.log(np.abs(x))))

def bearingEnergy(Y, df, X, bearing):
    """
    Energy within band of typical characteristic frequencies

    Parameters
    ----------
    Y : float 1D array
        Spectrum aplitude
    df : float
        Frequncy spacing in Hz
    X : float
        Shaft speed in Hz
    bearing : float 1D array
        Bearing characteristic frequencies in orders (i.e. per revolution)
        bearing[0] - Inner race
        bearing[1] - 2x roller spin frequency
        bearing[2] - Cage frequency
        bearing[3] - Outer race frequency
    """

    lowerFrequency = np.min([bearing[0], bearing[1], bearing[3]])*X*0.95
    upperFrequency = np.max([bearing[0], bearing[1], bearing[3]])*X*1.05
    i1 = int(np.floor(lowerFrequency/df))
    i2 = int(np.ceil(upperFrequency/df))
    return np.sum(Y[i1:i2+1])

def snr(r, Fs, ma=0.05, mb=0.5, cb=3, mc=0.05, md=0.6, c=2, c_ech=0.05, J_min=3, toler=0.1):
    """Estimates the Signal-to-noise ratio.

    Based on "About periodicity and signal to noise ratio - The
    strength of the autocorrelation function."
    by Nadine Martin and Corinne Mailhes

    Parameters
    ----------
    r : float 1D array
        The signal to estimate SNR of
    Fs : float
        Sampling rate
    ma : float
        Lag support start, percentage of r.size
    mb : float
        Lag support end, percentage of r.size
    cb : int
        Tolerance factor
    mc : float
        Lag support start, percentage of r.size
    md : float
        Lag support end, percentage of r.size
    c : int
        Tolerance factor
    c_ech : float
        Tolerance factor
    J_min : int
        Minimum number of detected maxima
    toler : float
        Tolerance factor applied to median

    Returns
    -------
    SNR_hat : float
        Estimated SNR ratio in dB
    flags : list, size=4
        Extra information about the calculation

        Element[i]:
        0 : boolean
            0 if r is not aperiodic (Positive)
            1 if r is random noise
        1 : boolean
            0 if card(ksi) > J_min (Positive)
            1 else
        2 : float
            Confidence value ratio
        3 : float
            Estimated fundamental frequency
    """

    def autocorr(r):
        d = r.size
        Rr = fftwconvolve(np.flipud(r), r, 'full')
        Rr = Rr[d-1:Rr.size]/np.arange(d, 0, -1, dtype=float)
        return Rr

    flags = []
    N = r.size

    # Step 1 Compute maximum autocorrelation function
    Rr_hat = autocorr(r)
    ma = int(ma*N)
    mb = int(mb*N)
    m_max = np.argmax(Rr_hat[ma:mb])
    m_max += ma
    Rr_hat_max = Rr_hat[m_max]

    # Step 2 Set initial power, and check if Rr is associated with noise
    Px_hat_init = Rr_hat_max
    Pn_hat_init = Rr_hat[0] - Px_hat_init
    sigma_max = np.sqrt(float(N))/(np.abs(N - m_max))*Pn_hat_init
    if Rr_hat_max <= cb*sigma_max:
        flags.append(1)
        print('Negative: r[n] is random noise')
    else:
        flags.append(0)
        # print('Positive: r[n] not aperiodic')

    # Step 3 Compute the lag set ksi
    mc = int(mc*N)
    md = int(md*N)
    listnumber = 0
    ksi = np.zeros(md - mc)
    mlist = np.arange(mc, md)
    sigma = (4*N - 6*mlist)/((N - mlist)**2.0)*Pn_hat_init*Px_hat_init + N/((N - mlist)**2.0)*Pn_hat_init**2.0
    for mj in range(0, mlist.size):
        leftside = Px_hat_init - c*sigma[mj] - c_ech*Px_hat_init
        rightside = Px_hat_init + c*sigma[mj] - c_ech*Px_hat_init
        if leftside <= Rr_hat[mj] <= rightside:
            ksi[listnumber] = mlist[mj]
            listnumber += 1
    ksi = np.array(ksi[0:listnumber], dtype=int)
    J = ksi.size
    if J > J_min:
        flags.append(0)
        # print('card(ksi) is larger than Jmin')
    else:
        flags.append(1)
        print('card(ksi) is smaller than Jmin')

    if J > J_min:
        # Step 4 Evaluate the fundamental-periodicity component Cfun
        ksi_d = np.diff(ksi)
        Med_ksi_d = np.median(ksi_d)
        Cfun = 0
        for i in range(0, ksi_d.size):
            u = np.abs(ksi_d[i] - Med_ksi_d)
            if u < toler*Med_ksi_d:
                Cfun += 1
        Cfun /= ksi_d.size
        f_hat_fun = Fs/Med_ksi_d
        flags.append(Cfun)
        flags.append(f_hat_fun)
        # print('Confidence ratio is %i percentage' % (Cfun*100.0))
        # print('Estimated fundamental frequency is %f Hz' % (f_hat_fun))

        # Step 5 Get SNR
        if J > J_min:
            Px_hat = 1/J*np.sum(Rr_hat[ksi])
            Pn_hat = Rr_hat[0] - Px_hat
            SNR_hat = 10*np.log10(Px_hat/Pn_hat)
            # print('Estimated SNR is %.4f' % (SNR_hat))
    else:
        flags.append(0)
        flags.append(0)
        SNR_hat = 0.0

    return SNR_hat, flags

def maxToMinPowerDensityDrop(Y, df, X, bearing):
    """
    Maximum to minimum power density drop

    Parameters
    ----------
    Y : float 1D array
        Spectrum aplitude
    df : float
        Frequncy spacing in Hz
    X : float
        Shaft speed in Hz
    bearing : float 1D array
        Bearing characteristic frequencies in orders (i.e. per revolution)
        bearing[0] - Inner race
        bearing[1] - 2x roller spin frequency
        bearing[2] - Cage frequency
        bearing[3] - Outer race frequency
    """

    lowerFrequency = np.min([bearing[0], bearing[1], bearing[3]])*X*0.95
    upperFrequency = np.max([bearing[0], bearing[1], bearing[3]])*X*1.05
    i1 = int(np.floor(lowerFrequency/df))
    i2 = int(np.ceil(upperFrequency/df))
    I = int(X/df)
    if I == 0:
        I = 1
    cc = fftwconvolve(np.ones(I)/float(I), Y)
    return np.max(cc)/np.min(cc)
