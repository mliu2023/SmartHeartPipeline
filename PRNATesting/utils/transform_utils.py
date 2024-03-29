import numpy as np
from scipy.signal import resample, iirnotch, lfilter
from biosppy.signals.tools import filter_signal
from biosppy.signals.ecg import ecg

# composes a list of transforms together
# all transforms should take in an ndarray and return an ndarray
# every transform needs to have an __init__ and a __call__ method
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, ecg_signal, source_freq):
        for t in self.transforms:
            # if the transform is resample, then we have to pass in the source frequency as well
            if(isinstance(t, Resample) or isinstance(t, NotchFilter)):
                ecg_signal = t(ecg_signal, source_freq)
            else:
                ecg_signal = t(ecg_signal)
        return ecg_signal

# resamples the ecg signal to 500 Hz
class Resample(object):
    def __init__(self, new_freq = 500):
        self.new_freq = new_freq
    def __call__(self, ecg_signal, source_freq = 500):
        ecg_signal = resample(ecg_signal, int(len(ecg_signal[0]) * (self.new_freq / source_freq)), axis=1)
        return ecg_signal

# removes specific frequencies from the ecg signal
class NotchFilter(object):
    def __init__(self, remove_freqs = [50, 60]):
        self.remove_freqs = remove_freqs
    def __call__(self, ecg_signal, source_freq = 500):
        quality_factor = 20.0
        for freq in self.remove_freqs:
            b,a = iirnotch(freq, quality_factor, source_freq)
            ecg_signal = lfilter(b, a, ecg_signal)
        return ecg_signal

class FiniteImpulseResponse():
    pass

class Normalize(object):
    def __init__(self, type='min-max'):
        self.type = type

    def __call__(self, ecg_signal):
        # not 100% sure if this works
        if self.type == 'min-max':
            max = np.amax(ecg_signal, axis=1, keepdims=True)
            min = np.amin(ecg_signal, axis=1, keepdims=True)
            eps = 1e-8
            ecg_signal = (ecg_signal-min)/(max-min+eps)
        elif self.type == 'mean-std':
            mean = np.mean(ecg_signal, axis=1, keepdims=True)
            std = np.std(ecg_signal, axis=1, keepdims=True)
            ecg_signal = (ecg_signal-mean)/std
        elif self.type == 'none':
            ecg_signal = ecg_signal
        else:
            raise NameError(f'Normalization type {self.type} is not included.')
        return ecg_signal

# randomly crops a window of length crop_size from the signal
class RandomCropping(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size
    def __call__(self, ecg_signal):
        if(len(ecg_signal[0]) > self.crop_size):
            start = np.random.randint(0, len(ecg_signal[0])-self.crop_size)
            ecg_signal = ecg_signal[:,start:start+self.crop_size]
        
        return ecg_signal

# add zeros to the end of the signal to reach max_length
class ZeroPadding(object):
    def __init__(self, min_length, padtype='end'):
        self.padtype = padtype
        self.min_length = min_length

    def __call__(self, ecg_signal):
        if(len(ecg_signal[0]) < self.min_length):
            ecg_signal = np.pad(ecg_signal, ((0,0), (0, self.min_length-len(ecg_signal[0]))))
        return ecg_signal

# apply bandpass filter to signal
class BandFilter(object):
    def __init__(self, filter_bandwith):
        self.filter_bandwith = filter_bandwith
    
    def __call__(self, ecg_signal, fs=500):
        order = int(0.3 * self.fs)

        # Filter signal
        ecg_signal, _, _ = filter_signal(signal=ecg_signal,
                                     ftype='FIR',
                                     band='bandpass',
                                     order=order,
                                     frequency=self.filter_bandwidth,
                                     sampling_rate=fs)

        return ecg_signal

# applyg biosppy flitering
class BiosppyFilter(object):
    def __init__(self, new_freq = 500):
        self.new_freq = new_freq
    def __call__(self, ecg_signal):
        ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate = ecg(signal = ecg_signal, sampling_rate = self.new_freq)
        return np.array(filtered)