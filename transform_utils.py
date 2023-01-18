import numpy as np
from scipy.signal import resample, iirnotch, lfilter

# composes a list of transforms together
# all transforms should take in an ndarray and return an ndarray
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, ecg_signal, source_freq):
        for t in self.transforms:
            # if the transform is resample, then we have to pass in the source frequency as well
            if(isinstance(t, Resample)):
                ecg_signal = t(ecg_signal, source_freq)
            else:
                ecg_signal = t(ecg_signal)
        return ecg_signal

# resamples the ecg signal to 500 Hz
class Resample(object):
    def __init__(self, new_freq = 500):
        self.new_freq = new_freq
    def __call__(self, ecg_signal, source_freq = 500):
        ecg_signal = resample(ecg_signal, int(len(ecg_signal) * (self.new_freq / source_freq)), axis=1)
        return ecg_signal

# removes specific frequencies from the ecg signal
class Notch_filter(object):
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
        if self.type == 'min-max':
            max = np.amax(ecg_signal, axis=1)
            min = np.amin(ecg_signal, axis=1)
            ecg_signal = (ecg_signal-min)/(max-min)
        elif self.type == 'mean-std':
            mean = np.mean(ecg_signal, axis=1)
            std = np.std(ecg_signal, axis=1)
            ecg_signal = (ecg_signal-mean)/std
        elif self.type == 'none':
            ecg_signal = ecg_signal
        else:
            raise NameError(f'Normalization type {self.type} is not included.')
        return ecg_signal

class RandomCropping(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size
    def __call__(self, ecg_signal):
        if(self.crop_size < len(ecg_signal[0])):
            start = np.random.randint(0, len(ecg_signal[0])-self.crop_size)
            ecg_signal = ecg_signal[:][start:start+self.crop_size]
        else:
            return ecg_signal
#add zeros to the end of the signal to reach a desired length
class ZeroPadding(object):
    def __init__(self, padtype='end'):
        self.padtype = padtype
    def __call__(self, ecg_signal):
        return ecg_signal