import numpy as np
import pywt
import torch
from scipy.signal import resample, iirnotch, lfilter
from scipy import interpolate
from biosppy.signals.tools import filter_signal
from biosppy.signals.ecg import ecg, engzee_segmenter, extract_heartbeats

# composes a list of transforms together
# all transforms should take in an ndarray and return an ndarray
# every transform needs to have an __init__ and a __call__ method
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, ecg_signal, source_freq, rpeaks=None):
        for t in self.transforms:
            # if the transform is resample, then we have to pass in the source frequency as well
            if(isinstance(t, Resample) or isinstance(t, NotchFilter)):
                ecg_signal = t(ecg_signal, source_freq)
            elif(isinstance(t, RpeakCropping)):
                ecg_signal = t(ecg_signal, source_freq, rpeaks)
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

# crops a window of length crop_size from the signal
class Cropping(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size
    def __call__(self, ecg_signal):
        if(len(ecg_signal[0]) > self.crop_size):
            ecg_signal = ecg_signal[:,:self.crop_size]
        
        return ecg_signal

# add zeros to the end of the signal to reach min_length
class ZeroPadding(object):
    def __init__(self, min_length, padtype='end'):
        self.padtype = padtype
        self.min_length = min_length

    def __call__(self, ecg_signal):
        if(len(ecg_signal[0]) < self.min_length):
            ecg_signal = np.pad(ecg_signal, ((0,0), (0, self.min_length-len(ecg_signal[0]))))
        return ecg_signal
    
# add zeros to the end of the signal to reach min_length
class ZeroPaddingBothSides(object):
    def __init__(self, min_length, padtype='end'):
        self.padtype = padtype
        self.min_length = min_length

    def __call__(self, ecg_signal):
        if(len(ecg_signal[0]) < self.min_length):
            pad_length = self.min_length-len(ecg_signal[0])
            pad_right = pad_length // 2
            pad_left = pad_length - pad_right
            ecg_signal = np.pad(ecg_signal, ((0,0), (pad_left, pad_right)))
        return ecg_signal
    
# wavelet transform
# work in progress
class continuousWaveletTransform(object):
    def __init__(self,source_freq):
        # scales are the sizes (height) of the wavelets each time it passes through (the numbers can be changed based on the signal)
        self.scales = np.arange(1,64)
        # wavelet type can be changed, basically changes the shape of the wavelet which parses the signal
        self.wavelet = 'morl'
        # since the array needs to be fed into the net, it can be resized to be like the original ECG
        # this var is the method
        self.interpolation = 'cubic'
    
    def __call__(self, ecg_signal, source_freq, rpeaks):
        
        # using the continuous wavelet transform
        coefficients, frequencies = pywt.cwt(ecg_signal, self.scales, self.wavelet)

        # using the numpy arrays to make the signal like the ecg signal, 
        coefficient_resizing = interpolate.interp2d(np.arange(coefficients.shape[1]), np.arange(coefficients.shape[0]), coefficients, kind = self.interpolation)(np.arange(coefficients.shape[1]), np.arange(ecg_signal.shape[0]))
        
        # currently just normalized it like this but its prob better to use the function at the top
        normalized_coefficients = (coefficient_resizing - np.min(coefficient_resizing)) / (np.max(coefficient_resizing) - np.min(coefficient_resizing))
        return normalized_coefficients



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

# get templates
class RpeakCropping(object):
    def __init__(self, rpeaks, new_freq = 500, crop_size = 7500):
        self.rpeaks = rpeaks
        self.new_freq = new_freq
        self.crop_size = crop_size
    def __call__(self, ecg_signal):
        signal = []
        for j in range(12):
            before = 150
            after = 300
            R = np.sort(self.rpeaks[j])
            rpeaks = [r for r in R if r >= before and r <= self.cropsize-after]
            s = ecg_signal[j][rpeaks[0]-before:rpeaks[-1]+after]
            s = np.pad(s, (0,self.crop_size-len(s)))
            signal.append(s)
        signal = torch.stack(signal)
        return ecg_signal
    
# denoising using discrete wavelet transform
class DWT(object):
    def __init__(self, wavelet, level):
        self.wavelet = wavelet
        self.level = level
    
    def __call__(self, ecg_signal):
        ecg_signal = wavelet_denoising(ecg_signal, wavelet=self.wavelet, level=self.level)
        return ecg_signal

def madev(d, axis=None):
    """ Mean absolute deviation of a signal """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)
    
def wavelet_denoising(x, wavelet='sym7', level=3):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * madev(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='soft') for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='per')

class FibrWavelet(object):
    def __init__(self):
        self.filter_bank = [[0.001590, -0.056193, 0.056736, 0.493436],
                            [0.493436, 0.056736, -0.056193, 0.001590],
                            [-0.493436, 0.056736, 0.056193, 0.001590],
                            [0.001590, 0.056193, 0.056736, -0.493436]]
    def get_wavelet(self):
        wavelet = pywt.Wavelet('Fibr Wavelet', filter_bank=self.filter_bank)
        print(wavelet)
        return wavelet