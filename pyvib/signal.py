"""
Signal processing algorithms
"""

from math import exp, pi, sqrt

import numpy as np
import pyfftw
from numba import njit
from psutil import cpu_count
from scipy import fftpack
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d

nthreads = cpu_count(logical=False)
pyfftw.interfaces.cache.enable()

def envelope(y, absolute=True):
    """
    Get envelope of signal.

    Parameters
    ----------
    y : float 1D array
        Signal to get envelope of
    absolute : boolean, optional
        Whether the absolute value (envelope is returned):

        - True for envelope
        - False for analytic signal

    Returns
    -------
    env : float 1D array
        Enveloped or analytic signal

    See Also
    --------
    fftwhilbert : function
        A faster way of calculating the analytic signal
    """

    ya = _fftwhilbert(y)
    if absolute is True:
        return np.abs(ya)
    else:
        return ya

def fftwconvolve(in1, in2, mode="full"):
    """
    Convolve two N-dimensional arrays using PYFFTW.
    This is a modified version of scipy.signal.fftwconvolve

    Convolve `in1` and `in2` using the fast Fourier transform method, with
    the output size determined by the `mode` argument.

    Parameters
    ----------
    in1 : float 1D array
        First input.
    in2 : float 1D array
        Second input. Should have the same number of dimensions as `in1`.
        If operating in 'valid' mode, either `in1` or `in2` must be
        at least as large as the other in every dimension.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:

        ``full``
           The output is the full discrete linear convolution
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.

    Returns
    -------
    out : float 1D array
        An N-dimensional array containing a subset of the discrete linear
        convolution of `in1` with `in2`.
    """

    def _centered(arr, newshape):
        # Return the center newshape portion of the array.
        newshape = np.asarray(newshape)
        currshape = np.array(arr.shape)
        startind = (currshape - newshape) // 2
        endind = startind + newshape
        myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
        return arr[tuple(myslice)]

    in1 = np.array(in1, copy=True)
    in2 = np.array(in2, copy=True)

    if in1.ndim == in2.ndim == 0:  # scalar inputs
        return in1 * in2
    elif not in1.ndim == in2.ndim:
        raise ValueError("in1 and in2 should have the same dimensionality")
    elif in1.size == 0 or in2.size == 0:  # empty arrays
        return np.array([])

    s1 = np.array(in1.shape)
    s2 = np.array(in2.shape)
    complex_result = (in1.dtype.type == np.complex128 or
                      in2.dtype.type == np.complex128)
    shape = s1 + s2 - 1

    # Check that input sizes are compatible with 'valid' mode
    if mode=='valid':
        assert in1.size >= in2.size

    fshape = [fftpack.helper.next_fast_len(int(d)) for d in shape]
    fslice = tuple([slice(0, int(sz)) for sz in shape])
    if not complex_result:
        obj1 = pyfftw.builders.rfftn(in1, fshape, threads=nthreads, planner_effort='FFTW_ESTIMATE')
        sp1 = obj1()
        obj2 = pyfftw.builders.rfftn(in2, fshape, threads=nthreads, planner_effort='FFTW_ESTIMATE')
        sp2 = obj2()
        obj3 = pyfftw.builders.irfftn(sp1 * sp2, fshape, threads=nthreads, planner_effort='FFTW_ESTIMATE')
        ret = (obj3())[fslice].copy()
    else:
        # If we're here, it's because we need a complex transform
        obj1 = pyfftw.builders.fftn(in1, fshape, threads=nthreads, planner_effort='FFTW_ESTIMATE')
        sp1 = obj1()
        obj2 = pyfftw.builders.fftn(in2, fshape, threads=nthreads, planner_effort='FFTW_ESTIMATE')
        sp2 = obj2()
        obj3 = pyfftw.builders.ifftn(sp1 * sp2, fshape, threads=nthreads, planner_effort='FFTW_ESTIMATE')
        ret = (obj3())[fslice].copy()

    if mode == "full":
        return ret
    elif mode == "same":
        return _centered(ret, s1)
    elif mode == "valid":
        return _centered(ret, s1 - s2 + 1)
    else:
        raise ValueError("Acceptable mode flags are 'valid',"
                         " 'same', or 'full'.")

def _fftwhilbert(x, N=None, axis=-1):
    """
    Compute the analytic signal, using the Hilbert transform.

    The transformation is done along the last axis by default.

    Parameters
    ----------
    x : float 1D array
        Signal data.  Must be real.
    N : int, optional
        Number of Fourier components.  Default: ``x.shape[axis]``
    axis : int, optional
        Axis along which to do the transformation.  Default: -1.

    Returns
    -------
    xa : ndarray
        Analytic signal of `x`, of each 1-D array along `axis`
    """

    x = np.asarray(x)
    if np.iscomplexobj(x):
        raise ValueError("x must be real.")
    if N is None:
        N = x.shape[axis]
    if N <= 0:
        raise ValueError("N must be positive.")

    obj1 = pyfftw.builders.fft(x, N, axis=axis, threads=nthreads, planner_effort='FFTW_ESTIMATE')
    Xf = obj1()
    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2

    if x.ndim > 1:
        ind = [np.newaxis] * x.ndim
        ind[axis] = slice(None)
        h = h[ind]
    obj2 = pyfftw.builders.ifft(Xf*h, axis=axis, threads=nthreads, planner_effort='FFTW_ESTIMATE')
    x = obj2()
    return x

def ordertrack(t, y, t_s, s, ds, cubic=True):
    """
	Order track the vibration signal to match a desired position vector

	Parameters
	----------
	t : float 1D array
        Time signal
    y : float 1D array
        Signal to order track
    t_s : float 1D array
        Time of position signal
    s : float 1D array
        Position signal in number of rounds
    ds : float
        Desired delta position
    cubic : boolean, optional
        Whether cubic interpolation is used

	Returns
	-------
    s_ot : float 1D array
        Shaft position of returned signal
    y_ot : float 1D array
        Order tracked signal
	"""

    # Copy data
    t = np.array(t)
    y = np.array(y)
    t_s = np.array(t_s)
    s = np.array(s)

    # Parameters
    dt = t[1] - t[0]

    # Get the overlapping parts between y and s
    temp = t[0]
    t = t - temp
    t_s = t_s - temp
    i1 = 0
    i2 = t.size
    if t_s[0] > 0.0:
        i1 = int(np.ceil(t_s[0]/dt))
    if t_s[-1] < t[-1]:
        i2 = int(np.floor(t_s[-1]/dt))
    t = t[i1:i2]
    y = y[i1:i2]

    #Interpolate the measured position to match vibration time
    f = interp1d(t_s, s)
    s = f(t)

    #Make desired position signal
    s_ot = np.arange(s[0], s[-1], ds)

    # print('s.size=%i, y.size=%i' % (s.size, y.size))

    #Make order tracked signal
    if cubic is True:
        f = InterpolatedUnivariateSpline(s, y, k = 3)
    else:
        f = interp1d(s, y)
    y_ot = f(s_ot)

    #Return data
    return s_ot, y_ot

def _tsa(y, Fs, n):
    """
	Get the Time synchronous average of a signal

	Parameters
	----------
	y : float 1D array
        Signal
    Fs : float
        Sampling frequency
    n : int
        Number of times to split the signal

	Returns
	-------
    t_tsa : float 1D array
        Time of synchronized signal
    y_tsa : flaot 1D array
        Synchronous signal
	"""

    n = np.float(n)
    k = int(y.size/n)
    yTSA = np.zeros(k, dtype = y.dtype)
    for i in range(0, int(n)):
        yTSA += y[int(i*k):int((i+1)*k)]
    yTSA = yTSA/n
    dt = 1.0/Fs
    tTSA = np.arange(0, k)*dt
    return tTSA, yTSA

def tsaremove(y, Fs, X, debug=False):
    """
	Subtracts the synchronous average of a signal

	Parameters
	----------
    y : float 1D array
        Signal
    Fs : float
        Sampling rate in Hz
    X : float
        Synchronous speed in Hz

	Returns
	-------
    y_nonsync : float 1D array
        Non-synchronous signal
	"""

    rounds = int(np.floor(y.size/Fs*X))
    n = int(float(rounds)*Fs/X)
    t_s, y_s = _tsa(y[:n], Fs, rounds)
    ns = y_s.size
    y_as = np.copy(y)
    k = int(np.ceil(float(y.size)/ns))
    for i in range(0, k):
        q1 = i*ns
        q2 = (i+1)*ns
        if q2 > y.size:
            q2 = y.size
        y_as[q1:q2] -= y_s[:q2-q1]
    return y_as

def tsakeep(y, Fs, X, debug=False):
    """
	Duplicates the synchronous signal to the original length

	Parameters
	----------
    y : float 1D array
        Signal
    Fs : float
        Sampling rate in Hz
    X : float
        Synchronous speed in Hz

	Returns
	-------
    y_sync : float 1D array
        Synchronous signal
	"""

    rounds = int(np.floor(y.size/Fs*X))
    n = int(float(rounds)*Fs/X)
    t_s, y_s = _tsa(y[:n], Fs, rounds)
    ns = y_s.size
    y_sync = np.zeros(y.size)
    k = int(np.ceil(float(y.size)/ns))
    for i in range(0, k):
        q1 = i*ns
        q2 = (i+1)*ns
        if q2 > y.size:
            q2 = y.size
        y_sync[q1:q2] = y_s[:q2-q1]
    return y_sync

def _getregime(meanv, speedregimes):
    """
	Checks if a mean speed is within a regime. Used in gsakeep()

	Parameters
	----------
    meanv : float
        Mean shaft speed
    speedregimes : float 1D array
        Defined speed regimes

	Returns
	-------
    chosenregime : int
        Index determining which regime the mean speed belongs to
	"""

    return np.argmax(np.greater(meanv, speedregimes[:-1]) * np.less(meanv, speedregimes[1:]))

def generalized_synchronous_average(t, vib, t_s, s, R, ds=None):
    """
	Estimate the synchronous average of a signal that actually varies in speed
    Based on
    Abboud, D., Antoni, J., Sieg-Zieba, S., & Eltabach, M. (2016).
    Deterministic-random separation in nonstationary regime.
    Journal of Sound and Vibration, 362, 305-326.

	Parameters
	----------
    t : float 1D array
        Time of signal
    y : float
        Signal
    t_s : float 1D array
        Time of shaft position
    s : float
        Shaft position
    R : int
        Number of regimes
    ds : float, optional
        Force a certain delta position if wanted

	Returns
	-------
    gsa : float 1D array
        Synchronous signal which can be removed
    s_ot : float 1D array
        Order tracked position
    vib_ot : float 1D array
        Order tracked signal
	"""

    # Velocity
    Fs_s = 1.0/(t_s[1] - t_s[0])
    v = np.zeros(s.size)
    v[0:-1] = np.diff(s)*Fs_s
    v[-1] = v[-2]
    X = np.min(v)

    # Order Track
    Fs = 1.0/(t[1] - t[0])
    if ds is None:
        ds = (np.round(Fs/X))**-1.0
    XOT = ds*Fs
    s_ot, vib_ot = ordertrack(t, vib, t_s, s, ds, cubic=True)
    sot_v, v_ot = ordertrack(t_s, v, t_s, s, ds, cubic=True)

    # If v_ot gets larger than vib_ot, do some decimation
    if v_ot.size > vib_ot.size:
        v_ot = v_ot[0:vib_ot.size]
        sot_v = sot_v[0:vib_ot.size]

    # Set up number of cycles
    ncycles = int(v_ot.size*ds)
    cyclesize = int(np.round(ds**-1.0))

    # Set up speed regimes
    speedregimes = np.linspace(np.min(v_ot), np.max(v_ot), R+1)

    # Initialize SA for each regime
    sa = np.zeros((R, cyclesize))
    saN = np.zeros(R)

    # Chosen region for each cycle, and for the partial last one (hence +1)
    chosenregime = np.zeros(ncycles + 1, dtype=int)

    # Get synchronous average for each regime
    for i in range(0, ncycles + 1):
        i1 = i*cyclesize
        i2 = (i+1)*cyclesize
        meanv = np.mean(v_ot[i1:i2])
        currentregime = _getregime(meanv, speedregimes)
        if i < ncycles:
            sa[currentregime, :] += vib_ot[i1:i2]
            saN[currentregime] += 1.0
        chosenregime[i] = currentregime

    # Apply average for each regime
    for i in range(0, R):
        if saN[i] > 1:
            sa[i, :] /= saN[i]

    # Interpolation (smoothening) using a Gaussian kernel
    lambd = speedregimes[1] - speedregimes[0] # Recommended by paper

    # Get center speed of all regimes
    regime_center_speed = np.zeros(len(sa))
    for i in range(0, len(sa)):
        regime_center_speed[i] = (speedregimes[i] + speedregimes[i+1])/2.0

    # Get the GSA
    gsa = _get_gsa(sa, vib_ot.size, chosenregime.size, cyclesize, v_ot, regime_center_speed, lambd) #This one works!

    return gsa, s_ot, vib_ot

def autocorrelation(y, normalize=False):
    """
	Get auto-correlation of the signal

	Parameters
	----------
    y : float 1D array
        Signal
    normalize : boolean, optional
        Whether signal should be normalized

	Returns
	-------
	ac : float 1D array
        Autocorrelated signal
	"""

    assert y.dtype == np.float64

    ac = fftwconvolve(np.flipud(y), y, 'full')
    if normalize:
        d = y.size
        ac[0:d] = ac[0:d]/np.arange(1, d+1, dtype = float)
        ac[d:ac.size] = ac[d:ac.size]/np.arange(d-1, 0, -1, dtype = float)

    return ac

def downsample(s, n, phase=0):
    """
    Direct downsampling of a signal

    Parameters
    ----------
    s : 1D array
        Signal to downsample
    n : int
        Downsampling factor
    phase : int
        Phase lag before sampling

    Returns
    -------
    s_out : 1D array
        Downsampled signal
    """

    return s[phase::n]

def teager(vib):
    """
    Compute the teager energy operator

    Parameters
    ----------
    x : 1D array
        Input signal

    Returns
    -------
    x_teager : float 1D array
        Teager energy operator
    """

    return vib[1:-1]*vib[1:-1] - vib[:-2]*vib[2:]

@njit(cache=True)
def encoder2position(t, enc, thr):
    """
    Converts from encoder pulses to position signal

    Parameters
    ----------
    t : float 1D array
        Time vector
    enc : float 1D array
        Measured analog signal
    thr : float
        Threshold for pulses going up or down

    Returns
    -------
    t_s : float 1D array
        Position time vector
    s : float 1D array
        Position vector
    n : int
        Number of pulses encountered
    """

    s = np.zeros(enc.size)
    t_s = np.zeros(enc.size)
    i = 0
    n = 0

    for i in range(0, enc.size - 1):
        if enc[i+1] >= thr and enc[i] < thr:
            n+= 1
            s[n] = s[n-1] + 1
            t_s[n] = t[i+1]
    return t_s, s, n+1


@njit(cache=True)
def _kern(x):
    """
    Kernel for _get_gsa
    """

    K = 1.0/sqrt(2*pi)*exp(-0.5*x**2)
    return K

@njit(cache=True)
def _get_gsa(sa, vibot_size, chosenregime_size, cyclesize,
    v_ot, regime_center_speed, lambd):
    """
    Jit function for GSA
    """

    # Initialize stuff
    gsa = np.zeros(vibot_size)
    kerncalcs = np.zeros(regime_center_speed.size)

    # To make it similar to c++ function;
    # sa = sa.ravel()

    # Loops
    for k in range(0, chosenregime_size):
        i1 = int(k*cyclesize)
        i2 = int((k+1)*cyclesize)
        if i2 > vibot_size:
            i2 = vibot_size
        iterrange = i2 - i1
        for i in range(0, iterrange):
            for j in range(0, regime_center_speed.size):
                kerncalcs[j] = _kern((v_ot[i1+i] - regime_center_speed[j])/lambd)
            rightsum = np.sum(kerncalcs[0:regime_center_speed.size]*sa[0:regime_center_speed.size, i])
            kerncalcs_sum = np.sum(kerncalcs)
            gsa[i1+i] = 1.0/kerncalcs_sum*rightsum

    return gsa
