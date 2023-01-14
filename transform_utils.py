import torch
import numpy as np
from scipy.signal import resample, iirnotch, lfilter
from sklearn.preprocessing import MinMaxScaler

# resamples the ecg signal to 500 Hz
def resample(ecg_signal, original_freq):
    new_freq = 500
    ecg_signal = resample(ecg_signal, int(len(ecg_signal) * (new_freq / original_freq)))
    return ecg_signal

# removes specific frequencies from the ecg signal
def notch_filter(ecg_signal, original_freq):
    remove_freqs = [50, 60]
    quality_factor = 20.0
    for freq in remove_freqs:
        b,a = iirnotch(freq, quality_factor, original_freq)
        ecg_signal = lfilter(b, a, ecg_signal)
    return ecg_signal

def fir():
    pass

def normalize(ecg_signal, original_freq):
    min = np.amin(ecg_signal, axis=1)
    max = np.amax(ecg_signal, axis=1)
    ecg_signal = (ecg_signal-min)/(max-min)
    return ecg_signal

