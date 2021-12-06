import numpy as np
import matplotlib.pyplot as plt
import pyvib

def main():
    # Simulate some vibration data
    fs = 250.0 / 60.0  # Reference shaft frequency [Hz]
    Fs = 51200.0  # Vibration sampling rate [Hz]
    Fs_s = Fs  # Position sampling rate [Hz]
    t = np.arange(0, 60.0, step=1/Fs)  # Time array [s]
    v = fs + fs*0.1*np.sin(t)  # Make shaft velocity with disturbance [rev/s]
    s = np.cumsum(v)/Fs  # Integrate shaft velocity [rev]
    fo = 5.12  # Fault order (number of bearing fault impacts per revolution) [orders]
    vib = pyvib.bearing.bearingsimulation(t, s, fo)  # Simulated vibration signal in [m/s2]

    # Perform order tracking to remove frequency variations
    ds = 1.0/(np.round(Fs/fs))  # Desired delta shaft position between each order tracked vibration sample
    s_ot, vib_ot = pyvib.signal.ordertrack(t, vib, t, s, ds)

    # Remove synchronous components with time synchronous average
    vib_ot = pyvib.signal.tsaremove(vib_ot, Fs, fs)

    # Make an envelope of the signal
    env = pyvib.signal.envelope(vib_ot)

    # FFT of the envelope
    Y, df = pyvib.fft.fft(env, Fs)

    # Plot the envelope FFT as a function of orders (1 is shaft speed, 2 is 2x shaft speed etc.)'
    # In the envelope spectrum, you can see harmonics of the bearing fault order
    do = df/fs
    fig, ax = plt.subplots()
    pyvib.plt.plotfft(Y, do, ax, xlim=[0, 30])
    ax.set_xlabel('Shaft orders')
    ax.set_ylabel('Envelope FFT Amplitude')
    ax.set_title('Envelope Spectrum')
    fig.tight_layout()

if __name__ == '__main__':
    main()
    plt.show()
