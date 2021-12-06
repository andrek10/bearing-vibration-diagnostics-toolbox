import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from pathlib import Path
import pyvib

def load_test_data(filename):
    """Load the NetCDF4 test file

    Parameters
    ----------
    filename : string
        Name of the file

    Returns
    -------
    vib : 1D array
        Vibration data
    Fs : float
        Vibration sampling rate
    s : 1D array
        Shaft position data given in number of revolutions
    Fs_s : float
        Sampling rate of position data
    totalrevs : float
        Number of revolutions the bearing has undergone before starting measurement
    """

    # Assuming that the test file is in the same folder
    path = Path('.').absolute()
    path_file = path.joinpath(filename)

    # Read the file
    rootgrp = Dataset(path_file, 'r')

    # Get the scale 
    vibunitscale = rootgrp['vib'].unitscale
    tachunitscale = rootgrp['tach'].unitscale

    # Set vibration and tachometer data as arrays
    vib = np.array(rootgrp['vib'][:], dtype=np.float64)*vibunitscale
    s = np.array(rootgrp['tach'][:])*tachunitscale

    # Get the sampling rates
    Fs = rootgrp['vib'].Fs
    Fs_s = rootgrp['tach'].Fs

    # Get the total number of revolutions
    totalrevs = rootgrp.totalrevs

    return vib, Fs, s, Fs_s, totalrevs

def main():
    vib, Fs, s, Fs_s, totalrevs = load_test_data("y2016-m09-d26-13-37-20.nc")

    # Set reference shaft speed in Hz
    fs_ref = 250.0/60.0

    # Make time vectors from sample rate
    t = np.arange(0, vib.size)/Fs
    t_s = np.arange(0, s.size)/Fs_s

    # Perform order tracking to remove frequency variations
    ds = 1.0/(np.round(Fs/fs_ref))  # Desired delta shaft position between each order tracked vibration sample
    s_ot, vib_ot = pyvib.signal.ordertrack(t, vib, t_s, s, ds)

    # Remove synchronous components with time synchronous average
    vib_ot = pyvib.signal.tsaremove(vib_ot, Fs, fs_ref)

    # Make an envelope of the signal
    env = pyvib.signal.envelope(vib_ot)

    # FFT of the envelope
    Y, df = pyvib.fft.fft(env, Fs)

    # Plot the envelope FFT as a function of orders (1 is shaft speed, 2 is 2x shaft speed etc.)'
    do = df/fs_ref
    fig, ax = pyvib.plt.figure(1, 1)
    pyvib.plt.plotfft(Y, do, ax, xlim=[0, 30])

if __name__ == '__main__':
    main()
    plt.show()
