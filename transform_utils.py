import torch
import numpy as np
from scipy.signal import resample, iirnotch, lfilter
from sklearn.preprocessing import MinMaxScaler

def resample(ecg_signal, original_freq):
    new_freq = 500
    resample(ecg_signal, int(len(ecg_signal) * (new_freq / original_freq)))

def notch_filter(ecg_signal, original_freq):
    remove_freqs = [50, 60]
    quality_factor = 20.0
    for freq in remove_freqs:
        b,a = iirnotch(freq, quality_factor, original_freq)
        ecg_signal = lfilter(b, a, ecg_signal)
def fir():
    pass

def normalize():
    pass

