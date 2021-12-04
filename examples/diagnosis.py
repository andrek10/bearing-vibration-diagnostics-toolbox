import numpy as np
from netCDF4 import Dataset
from pathlib import Path
import pyvib

def load_test_data(filename):
    # Assuming that the test file, i.e., y2016-m09-d26-13-37-20.nc, is in the same folder
    path = Path('.').absolute()
    path_file = path.joinpath(filename)

    # Read the file
    rootgrp = Dataset(path_file, 'r')

    # Get the scale 
    vibunitscale = rootgrp['vib'].unitscale
    tachunitscale = rootgrp['tach'].unitscale

    # Set vibration and tachometer data as arrays
    vib = np.array(rootgrp['vib'][:], dtype=np.float64)*vibunitscale
    s = np.array(rootgrp['tach'][:], dtype=np.float64)*tachunitscale

    # Get the sampling rates
    Fs = rootgrp['vib'].Fs
    Fs_s = rootgrp['tach'].Fs

    # Get the total number of revolutions
    totalrevs = rootgrp.totalrevs

    return vib, Fs, s, Fs_s, totalrevs

def main():
    vib, Fs, s, Fs_s, totalrevs = load_test_data("y2016-m09-d26-13-37-20.nc")

    # Perform order tracking to remove frequency variations
    t = np.arange(0, vib.size)/Fs
    t_s = np.arange(0, s.size)/Fs
    fs_ref = 250.0/60.0
    ds = 1.0/(np.round(Fs/fs_ref))
    s_ot, vib_ot = pyvib.signal.ordertrack(t, vib, t_s, s, ds)

    # Remove synchronous components with time synchronous average
    vib_ot = pyvib.signal.tsaremove(vib_ot, Fs, )


if __name__ == '__main__':
    main()
