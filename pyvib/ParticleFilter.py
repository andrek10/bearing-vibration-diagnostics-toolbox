"""
Particle filter class
"""

import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange

from .stats import percentile


def _weightedMean(particles, weights):
    """
    Weighted mean of particles
    """

    temp = 0.0
    temp2 = 0.0
    for i in prange(particles.shape[0]):
        temp += particles[i]*weights[i]
        temp2 += weights[i]
    temp2 += 1.e-300      # avoid round-off to zero
    if temp2 >= 1e-300:
        return temp/temp2
    else:
        return 0.0

@njit(cache=True)
def _weightedVar(mean, particles, weights):
    """
    Weighted variance of particles
    """

    temp = 0.0
    temp2 = 0.0
    for i in prange(particles.shape[0]):
        temp += (particles[i] - mean)**2*weights[i]
        temp2 += weights[i]
    temp2 += 1.e-300      # avoid round-off to zero
    if temp2 >= 1e-300:
        return temp/temp2
    else:
        return 0.0

@njit(cache=True)
def _systematicResample(weights, indexes, randomnumber):
    """
    Performs the systemic resampling algorithm used by particle filters.

    This algorithm separates the sample space into N divisions. A single random
    offset is used to to choose where to sample from for all divisions. This
    guarantees that every sample is exactly 1/N apart.

    Parameters
    ----------
    weights : list-like of float
        list of weights as floats
    """

    N = weights.shape[0]
    # make N subdivisions, and choose positions with a consistent random offset
    positions = (randomnumber + np.arange(N)) / N

    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1

def _normPdf(mu, var, z):
    """
    Gaussian normal PDF
    """

    return 1.0/((2*np.pi*var)**0.5)*np.exp(-(z - mu)**2/(2*var))

class ParticleFilter():
    """
    A particle filter class

    Parameters
    ----------
    N : int
        Number of particles
    R : float or array_like
        Variance of measured states
        len(R) == len(measuredStates)
    Q : float or array_like
        Variance of actuation error
        Part of model
    model : function(u, states, parameters, Q)
        Model that generates next step of states
        using previous states, parameters and Q
        statesDerivative can be used as a placeholder for the derivative
        Example:

        def model(u, states, parameters, statesDerivative, Q):
            m = parameters[:, 0]
            k = parameters[:, 1]
            c = parameters[:, 2]
            dt = 1.0
            statesDerivative[:, 0] = states[:, 1]
            statesDerivative[:, 1] = 1.0/m*(-k*states[:, 0] - c*states[:, 1] + (u + randn(states.shape[0])*np.sqrt(Q))
            states[:, 0] += statesDerivative[:, 0]*dt
            states[:, 1] += statesDerivative[:, 1]*dt

    nStates : int
        Number of states in the system
    nParameters : int
        Number of parameters in the system
    measuredStates : int or array_like
        Which state number are measured
        Could be a single number or multiple in a list.
        Observation (z) must have the same length.
    """

    def __init__(self, N, R, Q, model, nStates, nParameters, measuredStates, resampleAlways=False, resampleDebug=False):
        self.N = N
        if not type(R) == list and not type(R) == np.ndarray:
            self.R = [R]
        else:
            self.R = R
        if not type(Q) == list and not type(Q) == np.ndarray:
            self.Q = [Q]
        else:
            self.Q = Q
        self.model = model
        self.nStates = nStates
        self.nParameters = nParameters
        self.particles = np.empty((self.N, self.nParameters + self.nStates))
        self.weights = np.ones(self.N, float)/self.N
        self.indexes = np.zeros(self.N, 'i')
        self.mean = np.zeros(self.particles.shape[1])
        self.var = np.zeros(self.particles.shape[1])
        if type(measuredStates) is not list or type(measuredStates) is not np.ndarray:
            self.measuredStates = [measuredStates]
        else:
            self.measuredStates = measuredStates
        self.meanList = []
        self.varList = []
        self.iter = 0
        self.statesDerivative = np.empty((self.N, self.nStates))
        self.resampleIterations = []
        self.resampleAlways = resampleAlways
        self.resampleDebug = resampleDebug
        self.converged = False

    def createUniformParticles(self, ranges):
        """
        Create uniformly distributed particles

        Parameters
        ----------
        ranges : 2D numpy.ndarray
            The uniform range of starting guess
            Shaped as [nStates + nParamteres, 2]
        """

        self.createParticleParameters = ['uniform', ranges]
        for i in range(0, self.nParameters + self.nStates):
            self.particles[:, i] = np.random.uniform(ranges[i, 0], ranges[i, 1], size=self.N)
        self.weights = np.ones(self.N, float)/self.N
        self.converged = False

    def createGaussianParticles(self, mean, var):
        """
        Create gaussian distributed particles

        Parameters
        ----------
        mean : array_like
            Mean value of gaussian distributed guess
            len(mean) = nStates + nParameters
        std : array_like
            Variation of gaussian distributed guess
            len(var) = nStates + nParameters
        """

        self.createParticleParameters = ['gaussian', mean ,var]
        for i in range(0, self.nParameters + self.nStates):
            self.particles[:, i] = mean[i] + np.random.randn(self.N)*np.sqrt(var[i])
        self.weights = np.ones(self.N, float)/self.N
        self.converged = False

    def predict(self, u):
        """
        Predict state of next time step using control input

        Parameters
        ----------
        u : float or array_like
            The control input.
            Must follow rules of model function
        """

        self.iter += 1
        states = self.get_states()
        parameters = self.get_parameters()
        # update states
        self.model(u, states, parameters, self.Q, self.statesDerivative)

    def get_states(self):
        """
        Return the states of particles

        Returns
        -------
        states : float 2D array
            States of particles
        """

        return self.particles[:, 0:self.nStates]

    def get_parameters(self):
        """
        Return the parameters of particles

        Returns
        -------
        parameters : float 2D array
            Parameters of particles
        """

        return self.particles[:, self.nStates:self.nStates + self.nParameters]

    def update(self, z, debug=False):
        """
        Update the weights based on measurements and observation noise

        z : float or array_like:
            The observation
            len(z) == len(measuredStates)
        """

        if not type(z) == list or type(z) == np.ndarray:
            z = [z]
        # self.weights  .fill(1.)
        n = 0
        for i in self.measuredStates:
            self.weights *= _normPdf(self.particles[:, i], self.R[n], z[n])
            n += 1
        if self.weights.sum() < 1e-10:
            self.converged = True

        self.weights += 1.e-300      # avoid round-off to zero
        temp = self.weights.sum()
        if temp >= 1e-300:
            self.weights /= temp # normalize
        else:
            print('pf.update: weight sum is zero')

    def resample(self, thrScale=0.5):
        """
        Resamples particles IF necessary

        Parameters
        ----------
        thrScale : float, optional
            Thresholds for resampling scaled by number of particles
        """

        if not np.isnan(self.weights).any():
            if self._neff() < self.N*thrScale or self.resampleAlways is True:
                if self.resampleDebug:
                    print('Resamples at iter %i' % (self.iter))
                _systematicResample(self.weights, self.indexes, np.random.rand())
                self._resampleFromIndex()
                self.resampleIterations.append(self.iter)
                self.weights = np.ones(self.N)/self.N

    def estimate(self):
        """
        Estimates true value and variance of states and parameters
        Results are saved in ParticleFilter.meanList and -.varList
        """

        for i in range(0, self.particles.shape[1]):
            self.mean[i] = _weightedMean(self.particles[:, i], self.weights)
            self.var[i] = _weightedVar(self.mean[i], self.particles[:, i], self.weights)
        self.meanList.append(np.array(self.mean))
        self.varList.append(np.array(self.var))

    def getMeanAndVariance(self):
        """
        Get meanlist and varlist
        Mean and var

        Returns
        -------
        meanList : list of float 1D array
            Mean of each state for each time step
        varList : list of float 1D array
            Variance of each state for each time step
        """

        return np.array(self.meanList), np.array(self.varList)

    def getPercentile(self, per):
        """
        Get the percentile of values

        Parameters
        ----------
        per : float
            Percentile <0.0, 1.0>

        Returns
        -------
        percentile : float 1D array
            Percentiles
        """
        return np.array([np.percentile(np.array(self.meanList)[row,:], per) for row in range(0, len(self.meanList))])

    def plotHistogram(self, column):
        """
        Plot histogram of a state or parameter

        Parameters
        ----------
        column : int
            Which column of self.particles should be pltted
        """

        fig, ax = plt.subplots()
        ax.hist(self.particles[:, column], 100)

    def simulateTrend(self, iterations, u, percs=[0.5,]):
        """
        Simulate the trend moving forward using the particles parameters and state

        Parameters
        ----------
        iterations : int
            Number of iterations to simulate
        u : Data type defined by user in self.model
            Control input
        percs : list of floats, optional
            Which percentile of parameters and states to simulate as true values

        Returns
        -------
        output : float 3D array
            Simulation results
            output[i, j, k] gives the following
            i - the iteration
            j - the percentile
            k - the state number
        """

        particles = np.array(self.particles)
        statesDerivative = np.array(self.statesDerivative)
        parameters = particles[:, self.nStates:self.nStates + self.nParameters]
        output = np.empty((iterations, len(percs), self.nStates))
        weights = np.array(self.weights)
        for i in range(0, iterations):
            states = particles[:, 0:self.nStates]
            if type(u) == list or type(u) == np.ndarray:
                self.model(u[i], states, parameters, self.Q, statesDerivative)
            else:
                self.model(u, states, parameters, self.Q, statesDerivative)
            for k in range(0, self.nStates):
                # states[:, k] += np.random.randn(states[:, k].size)*sqrt(self.R[k])
                for j in range(0, len(percs)):
                    output[i, j, k] = percentile(states[:, k], percs[j], weights)
        return output

    def _neff(self):
        """
        The check to determine whether to re-sample

        Returns
        -------
        neff : float
        """

        return 1.0 / np.sum(np.square(self.weights))

    def _resampleFromIndex(self):
        """
        Performs resampling based on new indexes
        """

        self.particles[:] = self.particles[self.indexes]
        self.weights[:] = self.weights[self.indexes]
        self.weights.fill (1.0 / self.weights.shape[0])
