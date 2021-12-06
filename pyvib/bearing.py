"""
Functions related to i.e. bearing lifetime and geometry
"""

import numpy as np

from .misc import _XOR


class bearinglifetimemodel(object):
    """
    Makes a Weibull lifetime distribution for a bearing

	Parameters
	----------
    L10 : float
        Lifetime of bearing in million revolutions

    Methods
    -------
    life:
        Get lifetime in million revs based on probability of survival
    survival:
        Get survival probability based on number of revs

	Returns
	-------
	bearing : class object
        Bearing object
	"""

    def __init__(self, L10):
        self.L10 = L10
        self.L1 = self.L10/4.0
        self.s1 = np.log(np.log(1.0/0.99))
        self.s10 = np.log(np.log(1.0/0.9))
        
        self.a = self.L1**(self.s10/self.s1)
        self.b = 1.0/(self.s10/self.s1 - 1.0)
        self.Lbeta = (self.a/self.L10)**self.b
        self.e = self.s1/np.log(self.L1/self.Lbeta)
        
    def life(self, S):
        """
        Get lifetime in million revs based on probability of survival

        Parameters
        ----------
        S : float
            Probability of survival. <0.0, 1.0>

        Returns
        -------
        L : float
            Lifetime in million revolutions
        """

        s = np.log(np.log(1.0/S))
        L = self.Lbeta*np.exp(s/self.e)
        return L
        
    def survival(self, L):
        """
        Get survival probability based on number of revs

        Parameters
        ----------
        L : float
            Lifetime in million revs

        Returns
        -------
        S : float
            Survival probability <0.0, 1.0>
        """

        a = self.e*np.log(L/self.Lbeta)
        S = 1.0/(np.exp(np.exp(a)))
        return S
        
def bearingcharfrequencies(D, d, n, theta=0.0):
    """
    Calculate bearing characteristic orders from dimensions.
    Equations taken from:
    Rolling element bearing diagnostics-A tutorial by R Randall and J Antoni

	Parameters
	----------
    D : float
        Pitch diameter
    d : float
        roller diameter
    n : int
        Number of rollers
    theta : float, optional
        Contact angle in degrees

	Returns
	-------
	bearing : array_like
        Bearing fault orders (inner, roller, cage, outer)
	"""

    theta = theta*np.pi/180.0
    FTF = 1.0/2.0 * (1.0 - d/D*np.cos(theta))
    BPFO = n*FTF
    BPFI = n/2.0 * (1.0 + d/D*np.cos(theta))
    BSF = D/(2.0*d) * (1.0 - (d/D * np.cos(theta))**2.0)
    return np.array([BPFI, 2*BSF, FTF, BPFO])

def bearingsimulation(t, s, fo, beta=1000.0, omr=10400.0, taustd=0.02, initpause=0.0, amp=1.0, debug=False, seed=None):
    """
    Make a simulation of a bearing with an outer race fault

	Parameters
	----------
    t : float 1D array
        Time signal
    s : float 1D array
        Shaft angle position in number of rounds
    fo : float
        Outer race fault frequency
    beta : float, optional
        The damping ratio
    omr : float, optional
        Resonance frequency in Hz
    taustd : float, optional
        Standard deviation change in percentage between each impact
    initpause : float, optional
        Initial pause in rounds before first impact
    amp : float or function, optional
        The impact amplitude vibration:
        - If float, the same amplitude all the time
        - If function, the instantaneous velocity is used as input and
          the amplitude is returned
    debug : boolean, optional
        Whether debug information is returned
        If true, the percentage finished is printed
    seed : int or None, optional
        Choose a seed number for numpy, must be positive.

	Returns
	-------
    x : float 1D array
        The bearing vibration
	"""

    if seed is not None:
        np.random.seed(seed)
    omr = omr*2*np.pi
    dt = t[1] - t[0]
    Fs = 1.0/dt
    Thp = 1.0/fo
    M = int(s[-1]/Thp)
    #L = np.ones(M, dtype=float)*0.5
    tau = np.random.randn(M)*Thp*taustd
    #impactperiod = np.ones(M)*-1.0
    if callable(amp):
        v = np.diff(s)/dt
    x = np.zeros(t.size)
    impulselength = int(-np.log(0.005)/beta*Fs)
    per = 0.05
    for m in range(0, M, 1):
        sumtau = np.sum(tau[0:m])
        I = np.where(s > (m+initpause)*Thp + sumtau)[0]
        I = I[0:impulselength]
        impacttime = t[I[0]] 
        if callable(amp):
            L = amp(v[I[0]-1])
        else:
            L = amp
        p1 = L*np.exp(-beta*(t[I] - impacttime))
        p2 = np.sin(omr*(t[I] - impacttime))
        x[I] += p1*p2
        if float(m)/float(M) > per and debug is True:
            print('Finished %i percent' % (per*100))
            per += 0.05
    return x

def bearingsimulation_ahc(t, s, fo, amp, beta=1000.0, omr=3543.0, initpause=0.5):
    """
    Make a simulation of a bearing with an outer race fault
    where the shaft speed can pass 0 rpm speed.

	Parameters
	----------
    t : float 1D array
        Time signal
    s : float 1D array
        Shaft angle position in number of rounds
    fo : float
        Outer race fault frequency
    amp : function(v)
        Amplitude function with shaft velocity as input.
        Must return scalar amplitude only
    beta : float, optional
        The damping ratio
    omr : float, optional
        Resonance frequency in Hz
    initpause : float, optional
        Initial pause in fault-periods before first impact

	Returns
	-------
    x : float 1D array
        The bearing vibration
	"""

    # Make parameters
    omr = omr*2*np.pi
    dt = t[1] - t[0]
    Fs = 1.0/dt
    Thp = 1.0/fo
    v = np.diff(s)*Fs

    # Initialize vibratin
    x = np.zeros(t.size)

    # Check how long an impulse calculation should be
    impulselength = int(-np.log(0.0025)/beta*Fs)

    # Check when fault is struck
    sfault = np.array((s)*fo - initpause + 1, dtype=int) # -0.25 is cast to 0. +1 is added to cast 0.75 to 0 and 1.0 to 1
    XORresults = _XOR(sfault)
    Iimpact = np.where(XORresults==1)[0]

    # Make impulses on impact
    for j in range(0, len(Iimpact)):
        i = Iimpact[j]
        L = amp(v[i-1])
        p1 = L*np.exp(-beta*(t[i:i+impulselength] - t[i]))
        p2 = np.sin(omr*(t[i:i+impulselength] - t[i]))
        x[i:i+impulselength] += p1*p2

    return x
