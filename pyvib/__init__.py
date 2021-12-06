# ./__init__.py
import sys

from . import acoustics as acoustics
from . import bearing as bearing
from . import diagnose as diagnose,
from . import fft as fft
from . import filter as filter
from . import linalg as linalg
from . import misc as misc
from . import plt as plt
from . import signal as signal
from . import stats as stats
from . import ParticleFilter as ParticleFilter
from . import features as features

def set_seed(seed=None):
    for element in [acoustics, bearing, diagnose, fft,
    filter, linalg, misc, plt, signal, stats,
    ParticleFilter, features]:
        element.np.random.seed(seed)
