# ./__init__.py
import sys

from . import (acoustics, bearing, diagnose, fft,
    filter, linalg, misc, plt, signal, stats,
    ParticleFilter, features)

def set_seed(seed=None):
    for element in [acoustics, bearing, diagnose, fft,
    files, filter, linalg, load, misc, plt, signal, stats,
    ParticleFilter, features]:
        element.np.random.seed(seed)