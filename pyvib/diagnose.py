"""

"""

import numpy as np

from .fft import fft
from .linalg import get_SVDxi
from .signal import envelope, fftwconvolve


def R_SVD(U, sigma, V, time, f_fault, tolerance = 0.02, PMItreshold = 1.0, estimate_xi_func=get_SVDxi, estimate_xi_func_params=None):
    '''
    Get PMI and weights for each fault frequency in f_fault
    Based on the R-SVD algorithm in paper:
    "A novel strategy for signal denoising using reweighted SVD and
    its applications to weak fault feature enhancement of rotating
    machinery"

    Parameters
    ----------
    U : array 2D float
    sigma : array 1D float
        The singular values
    V : array 2D float
    time : array 1D float
        The time of the signal before SVD is computed
    f_fault : list or array_like
        Frequency of each fault type
    tolerance : float, optional
        Tolerance around f_fault that the peak is searched for.
    PMIthreshold : float, optional
        The treshold for which a PMI becomes a weight
    estimate_xi_func : function, optional
        Which function to use to estimate the X_i before autocorr and env.
        Deafult is based on an SVD in the time/angular domain.
        Must have (U, sigma, V, i) as input, and return x_i
    estimate_xi_func_params : list or None, optional
        List of extra variables to use as input
        None for no extra input

    Returns
    -------
    PMI : list of 1D array
        PMI for each fault period
    W : list of 1D array
        Wights for each fault period
    '''

    # Get the search region
    m = sigma.size
    f_fault = np.asanyarray(f_fault)
    dt = time[1] - time[0]
    T0 = np.zeros(f_fault.size, dtype=int)
    T1 = np.zeros(f_fault.size, dtype=int)
    PMI = []
    W = []
    for i in range(0, f_fault.size):
        T0[i] = int( np.floor( (1.0/(f_fault[i]*(1.0 + tolerance)))/dt ) )
        T1[i] = int( np.ceil( (1.0/(f_fault[i]*(1.0 - tolerance)))/dt ) )
        if T1[i] == T0[i]:
            T1[i] += 1
        PMI.append(np.zeros(m))
        W.append(np.zeros(m))

    # Calculate PMI for each fault type
    for i in range(0, m):
        if estimate_xi_func_params is None:
            a_i = estimate_xi_func(U, sigma, V, i)
        else:
            a_i = estimate_xi_func(U, sigma, V, i, estimate_xi_func_params)
        a_i = envelope(a_i)
        a_i -= a_i.mean()
        R_a = fftwconvolve(np.flipud(a_i), a_i)
        # Keep positive part
        R_a = R_a[a_i.size-1:]
        # Scale by dividing by number of elements
        R_a = R_a / np.arange(R_a.size, 0, -1)
        # Get T_0
        R_0 = R_a[0]
        # Calculate R and PMI for each fault type
        for k in range(0, f_fault.size):
            # print('T0[%i] = %f, T1[%i] = %f, R_a.size = %i' % (k, T0[k], k, T1[k], R_a.size))
            # print(R_a[481:501])
            R_T = np.max(R_a[T0[k]:T1[k]])
            PMI[k][i] = R_T/(R_0 - R_T)
    
    # Calculate weights
    for k in range(0, f_fault.size):
        temp = np.sum(PMI[k])
        for i in range(0, m):
            if PMI[k][i] > PMItreshold:
                W[k][i] = PMI[k][i]/temp
    
    # Return data
    return PMI, W

def ES_SVD(U, sigma, V, time, f_fault, f_side, PMItreshold, estimate_xi_func=get_SVDxi, estimate_xi_func_params=None):
    '''
    Envelope Score - SVD
    Get PMI and weights for each fault frequency in f_fault
    Based on envelope FFT score

    Parameters
    ----------
    U : array 2D float
    sigma : array 1D float
        The singular values
    V : array 2D float
    time : array 1D float
        The time of the signal before SVD is computed
    f_fault : list or array_like
        Frequency of each fault type
    f_side : list or array_like
        Side-band frequency for each fault type.
        Use 0.0 for entries without side-bands
    PMIthreshold : float, optional
        The treshold for which a PMI becomes a weight
    estimate_xi_func : function, optional
        Which function to use to estimate the X_i before env.
        Deafult is based on an SVD in the time/angular domain.
        Must have (U, sigma, V, i) as input, and return x_i
    estimate_xi_func_params : list or None, optional
        List of extra variables to use as input
        None for no extra input

    Returns
    -------
    PMI : list of 1D array
        PMI for each fault period
    W : list of 1D array
        Wights for each fault period
    '''

    # Get the search region
    m = sigma.size
    f_fault = np.asanyarray(f_fault)
    f_side = np.asanyarray(f_side)
    dt = time[1] - time[0]
    Fs = 1.0/dt
    PMI = [] #PMI is here the envelope score
    W = []
    for i in range(0, f_fault.size):
        PMI.append(np.zeros(m))
        W.append(np.zeros(m))

    # Calculate PMI for each fault type
    for i in range(0, m):
        if estimate_xi_func_params is None:
            a_i = estimate_xi_func(U, sigma, V, i)
        else:
            a_i = estimate_xi_func(U, sigma, V, i, estimate_xi_func_params)
        a_i = envelope(a_i)
        Y, df = fft(a_i, Fs)
        # Calculate PMI for each fault type
        for k in range(0, f_fault.size):
            PMI[k][i] = diagnosefft(Y, df, f_fault[k], 1.0, f_side[k])
    
    # Calculate weights
    for k in range(0, f_fault.size):
        temp = 0.0
        for i in range(0, m):
            if PMI[k][i] > PMItreshold:
                temp += PMI[k][i]
        for i in range(0, m):
            if PMI[k][i] > PMItreshold:
                W[k][i] = PMI[k][i]/temp
    
    # Return data
    return PMI, W

def diagnosefft(Y, df, charf, X, subband, debug=False, version=2, harmthreshold=3.0, subthreshold=3.0):
    """
	Diagnose a spectrum for bearing faults. Returns a score
	
	Parameters
	----------
    Y : float 1D array
        Spectrum values
    df : float
        Delta frequency in Hz
    charf : float
        Harmonic frequency
    X : float
        Shaft speed in Hz
    xubband : float
        Sideband frequency
    debug : boolean, optional
        Whether debug information is returned
    version : int, optional
        Which version of this script to run. Default 2 with new noise estimator
	
	Returns
	-------
	score : float
        Score for fault being present
	"""

    #Rescale fft
    df /= X
    nHarm = 1
    score = 0.0
    if debug is True:
        harmonics = []
        subbandsNeg = []
        subbandsPos = []
        noises = []
        scores = []
    while True:
        #if second harmonic, reduce the tolerance for finding peak
        if nHarm == 1:
            per = 0.02
        else:
            per = 0.01
        #Detect the charf harmonic
        j1 = int((nHarm*charf-per*charf)/df)
        j2 = int((nHarm*charf+per*charf)/df)
        jh = np.argmax(Y[j1:j2]) + j1
        harm = Y[jh]
        #Reclaibrate characteristic frequency
        charf = df*jh/nHarm
        #Detect the noise level for the harmonic
        if version == 1:
            j1n = int((nHarm*charf-0.02*charf)/df)
            j2n = int((nHarm*charf+0.02*charf)/df)
            if jh - j1n == 0:
                noise = np.mean(Y[jh+1:j2n])
            elif j2n - (jh + 1) == 0:
                noise = np.mean(Y[j1n:jh])
            else:
                noise = (np.mean(Y[j1n:jh]) + np.mean(Y[jh+1:j2n]))/2.0
        elif version == 2:
            # Find left bottom of harmonic
            for i in range(jh, 0, -1):
                if Y[i-1] > Y[i]:
                    jhl = i
                    break
            # Find right bottom of harmonic
            for i in range(jh, Y.size, 1):
                if Y[i+1] > Y[i]:
                    jhr = i
                    break
            # j1n = int((nHarm*charf-charf)/df)
            # j2n = int((nHarm*charf+charf)/df)
            noise = (np.mean(Y[jhl-2:jhl+1]) + np.mean(Y[jhr:jhr+3]))/2.0
        # print('j1=%i, j2=%i, jh=%i, harm=%f, jhl=%i, jhr=%i, noise=%f' % (j1,j2,jh,harm,jhl,jhr,noise))
        #If there should be subbands, detect them aswell
        if subband > 0.01:
            #Check for negative subband first
            j1 = int((nHarm*charf - subband - 0.05)/df)
            j2 = int((nHarm*charf - subband + 0.05)/df)
            jsn = np.argmax(Y[j1:j2]) + j1
            negSubBand = Y[jsn]
            #Check for position subband
            j1 = int((nHarm*charf + subband - 0.05)/df)
            j2 = int((nHarm*charf + subband + 0.05)/df)
            jsp = np.argmax(Y[j1:j2]) + j1
            posSubBand = Y[jsp]
        #Make the final score!
        #If subband should exist:
        if subband > 0.01:
            if harm >= noise*harmthreshold and (negSubBand > noise*subthreshold or posSubBand > noise*subthreshold):
                score += harm/(noise*3.0)*nHarm**2.0
                nHarm += 1
                if debug is True:
                    subbandsNeg.append(jsn)
                    subbandsPos.append(jsp)
                    harmonics.append(jh)
                    noises.append(noise)
                    scores.append(score)
            else:
                if debug is True:
                    return score, subbandsNeg, subbandsPos, harmonics, noises, scores
                else:
                    return score
        #if subband should not exist
        else:
            if harm >= noise*harmthreshold:
                score += harm/(noise*harmthreshold)*(nHarm+1.0)**2.0
                nHarm += 1
                if debug is True:
                    harmonics.append(jh)
                    noises.append(noise)
                    scores.append(score)
            else:
                if debug is True:
                    return score, subbandsNeg, subbandsPos, negSubBand, posSubBand, harmonics, noises, scores
                else:
                    return score
        
        #Check if FFT is too short. If so, return what is done!
        test1 = int((nHarm*charf+0.02*charf)/df)
        test2 = int((nHarm*charf + subband + 0.05)/df)
        if test1 > Y.size or test2 > Y.size:
            if debug is True:
                return score, subbandsNeg, subbandsPos, negSubBand, posSubBand, harmonics, noises, scores
            else:
                return score
                      
def diagnosevibrationfft(Y, df, X, bearing, radial=True):
    """
	Diagnose a spectrum for all three fault types
	
	Parameters
	----------
    Y : float 1D array
        Spectrum amplitude
    df : float
        Delta frequency in Hz
    X : float
        Shaft speed in Hz
    bearing : array_like of floats
        Bearing characteristic orders (inner,roller,cage,outer)
    radial : boolean, optional
        Whether radial forces exists. If so, sidebands exists
	
	Returns
	-------
    scores : float 1D array
        Score for [inner,roller,outer]
	"""
    score = np.zeros(3)
    f_c = np.zeros(3)
    f_sb = np.zeros(3)
    for i in range(0, 3):
        if i == 0: #Inner
            f_c[i] = bearing[0]
            if radial:
                f_sb[i] = 1.0
            else:
                f_sb[i] = 0.0
        elif i == 1: #Roller
            f_c[i] = bearing[1]
            if radial:
                f_sb[i] = bearing[2]
            else:
                f_sb[i] = 0.0
            # Cage
            
        elif i == 2: #Outer ring
            f_c[i] = bearing[3]
            f_sb[i] = 0.0
    
    for j in range(0, 3):
        tempScore = diagnosefft(Y, df, f_c[j], X, f_sb[j])
        score[j] += tempScore
                
    return score
