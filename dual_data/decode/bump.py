import numpy as np


def decode_bump(signal, axis=-1, windowSize=10, SMOOTH=False):
    signal_copy = signal.copy()
    if axis != -1 and signal.ndim != 1:
        signal_copy = np.swapaxes(signal_copy, axis, -1)

    if SMOOTH:
        signal_copy = circcvl(signal_copy, windowSize, axis=-1)

    length = signal_copy.shape[-1]
    dPhi = np.pi / length

    dft = np.dot(signal_copy, np.exp(-2.0j * np.arange(length) * dPhi))

    if axis != -1 and signal.ndim != 1:
        dft = np.swapaxes(dft, axis, -1)

    m1 = 2.0 * np.absolute(dft) / length
    phi = np.arctan2(dft.imag, dft.real) # % (2.0 * np.pi)
    
    return m1, phi


def circcvl(signal, windowSize=10, axis=-1):
    signal_copy = signal.copy()
    
    if axis != -1 and signal.ndim != 1:
        signal_copy = np.swapaxes(signal_copy, axis, -1)
    
    # Save the nan positions before replacing them
    nan_mask = np.isnan(signal_copy)
    signal_copy[nan_mask] = np.interp(np.flatnonzero(nan_mask), 
                                      np.flatnonzero(~nan_mask), 
                                      signal_copy[~nan_mask])
    
    ker = np.concatenate(
        (np.ones((windowSize,)), np.zeros((signal_copy.shape[-1] - windowSize,)))
    )

    smooth_signal = np.real(
        np.fft.ifft(
            np.fft.fft(signal_copy, axis=-1) * np.fft.fft(ker, axis=-1), axis=-1
        )
    ) * (1.0 / float(windowSize))
    
    # Substitute the original nan positions back into the result
    smooth_signal[nan_mask] = np.nan

    if axis != -1 and signal.ndim != 1:
        smooth_signal = np.swapaxes(smooth_signal, axis, -1)
    
    return smooth_signal
