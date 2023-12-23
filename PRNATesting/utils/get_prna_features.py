import numpy as np
import utils.feats.pyeeg as pyeeg
from utils.feats.utils.pyrem_univariate import *
from dataset import get_frequency_from_header
from biosppy.signals.ecg import ecg
from scipy.signal import resample
from pyentrp import entropy as ent
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def extract_features(ecg_signal, headerpath):
    #### process ecg signal ####
    new_freq = 500
    source_freq = get_frequency_from_header(headerpath)
    ecg_signal = resample(ecg_signal, int(len(ecg_signal) * (new_freq / source_freq)), axis=0)
    ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, instantaneous_heart_rate = ecg(signal = ecg_signal, sampling_rate = new_freq, show = False, interactive = False)

    #### extract general features ####
    rr_intervals = np.diff(rpeaks) # differences between peaks
    diff_rr_intervals = np.diff(rr_intervals) # differences between differences between peaks
    heart_rate = 60*new_freq/rr_intervals
    rpeak_amplitudes = filtered[rpeaks]
    
    #### heart rate features ####
    hr_min = np.amin(heart_rate)
    hr_max = np.amax(heart_rate)
    hr_mean = np.mean(heart_rate)
    hr_median = np.median(heart_rate)
    hr_std = np.std(heart_rate)
    hr_activity = np.mean(heart_rate**2)

    #### RR interval features ####
    rr_min = np.amin(rr_intervals)
    rr_max = np.amax(rr_intervals)
    rr_mean = np.mean(rr_intervals)
    rr_median = np.median(rr_intervals)
    rr_std = np.std(rr_intervals)
    rmssd = np.sqrt(np.mean(rr_intervals**2))

    # RR interval intra-cluster distance, cluster 1, 2, and 3
    rr_cluster_slope = 0
    rr_cluster_1 = 0
    rr_cluster_2 = 0
    rr_cluster_3 = 0

    if len(rr_intervals) > 6:
        rri_combined = np.column_stack((rr_intervals[0:-2], rr_intervals[1:-1])) # Combine r_ibi and r_ibi + 1
        clusters = range(1, 4) # Set cluster range
        cluster_models = [KMeans(n_clusters=cluster).fit(rri_combined) for cluster in clusters] # Compute KMeans clusters
        centroids = [cluster_model.cluster_centers_ for cluster_model in cluster_models] # Get centroids

        # Compute intra-cluster distances
        distances = [cdist(rri_combined, cent, 'euclidean') for cent in centroids]
        dist = [np.min(distance, axis=1) for distance in distances]
        ssd = [sum(d) / rri_combined.shape[0] * 1000 for d in dist]

        rr_cluster_slope = np.polyfit(x=clusters, y=ssd, deg=1)[0]
        rr_cluster_1 = ssd[0]
        rr_cluster_2 = ssd[1]
        rr_cluster_3 = ssd[2]

    # RR interval Fisher information
    rr_fisher = 0
    if len(heart_rate) > 1:
        de=1
        tau=2
        N = len(heart_rate)
        Y = np.zeros((de, N - (de - 1) * tau))
        for i in range(de):
            Y[i] = heart_rate[i *tau : i*tau + Y.shape[1]]
        mat = Y.T
        W = np.linalg.svd(mat, compute_uv = False)
        W /= sum(W)  # normalize singular values
        FI_v = (W[1:] - W[:-1])**2 / W[:-1]
        rr_fisher = np.sum(FI_v)

    #### diff RR interval features ####
    diff_rr_min = np.amin(diff_rr_intervals)
    diff_rr_max = np.amax(diff_rr_intervals)
    diff_rr_mean = np.mean(diff_rr_intervals)
    diff_rr_median = np.median(diff_rr_intervals)
    diff_rr_std = np.std(diff_rr_intervals)

    # pNN features
    pnn30 = sum(abs(diff_rr_intervals) > 0.03) / len(diff_rr_intervals) * 100
    pnn40 = sum(abs(diff_rr_intervals) > 0.04) / len(diff_rr_intervals) * 100
    pnn50 = sum(abs(diff_rr_intervals) > 0.05) / len(diff_rr_intervals) * 100
    pnn60 = sum(abs(diff_rr_intervals) > 0.06) / len(diff_rr_intervals) * 100
    pnn70 = sum(abs(diff_rr_intervals) > 0.07) / len(diff_rr_intervals) * 100
    pnn80 = sum(abs(diff_rr_intervals) > 0.08) / len(diff_rr_intervals) * 100
    pnn90 = sum(abs(diff_rr_intervals) > 0.09) / len(diff_rr_intervals) * 100

    #### amplitude features ####
    rpeak_min = np.amin(rpeak_amplitudes)
    rpeak_max = np.amax(rpeak_amplitudes)
    rpeak_mean = np.mean(rpeak_amplitudes)
    rpeak_median = np.median(rpeak_amplitudes)
    rpeak_std = np.std(rpeak_amplitudes)
    
    #### entropy features ####
    rpeak_ent = pyeeg.ap_entropy(rpeak_amplitudes, M=2, R=0.1*np.std(rpeak_amplitudes))

    # template_features = TemplateStatistics(ts, ecg_signal, filtered, rpeaks, templates, fs=new_freq)


    return np.array([hr_min, hr_max, hr_mean, hr_median, hr_std, hr_activity, 
                    rr_min, rr_max, rr_mean, rr_median, rr_std, rmssd, 
                    rr_cluster_slope, rr_cluster_1, rr_cluster_2, rr_cluster_3, 
                    rr_fisher, diff_rr_min, diff_rr_max, diff_rr_mean, diff_rr_median, diff_rr_std,
                    pnn30, pnn40, pnn50, pnn60, pnn70, pnn80, pnn90,
                    rpeak_min, rpeak_max, rpeak_mean, rpeak_median, rpeak_std,
                    rpeak_ent])

class TemplateStatistics:

    """
    Generate a dictionary of template statistics for one ECG signal.

    Parameters
    ----------
    ts : numpy array
        Full waveform time array.
    signal_raw : numpy array
        Raw full waveform.
    signal_filtered : numpy array
        Filtered full waveform.
    rpeaks : numpy array
        Array indices of R-Peaks
    templates_ts : numpy array
        Template waveform time array
    templates : numpy array
        Template waveforms
    fs : int, float
        Sampling frequency (Hz).
    template_before : float, seconds
            Time before R-Peak to start template.
    template_after : float, seconds
        Time after R-Peak to end template.

    Returns
    -------
    template_statistics : dictionary
        Template statistics.
    """

    def __init__(self, ts, signal_raw, signal_filtered, rpeaks,
                 templates, fs):

        # Input variables
        self.ts = ts
        self.signal_raw = signal_raw
        self.signal_filtered = signal_filtered
        self.rpeaks = rpeaks
        self.templates = templates
        self.fs = fs
        self.template_before_ts = 0.2
        self.template_after_ts = 0.4
        self.template_before_sp = int(self.template_before_ts * self.fs)
        self.template_after_sp = int(self.template_after_ts * self.fs)

        # Set QRS start and end points
        self.qrs_start_sp_manual = 30  # R - qrs_start_sp_manual
        self.qrs_end_sp_manual = 40  # R + qrs_start_sp_manual

        # R-Peak calculations
        self.template_rpeak_sp = self.template_before_sp

        # Calculate median template
        self.median_template = np.median(self.templates, axis=1)

        # Correct R-Peak picks
        # self.r_peak_check(correlation_threshold=0.9)

        # RR interval calculations
        self.rpeaks_ts = self.ts[self.rpeaks]

        # QRS calculations
        self.calculate_qrs_bounds()

        # PQRST Calculations
        self.preprocess_pqrst()

        # Feature dictionary
        self.template_statistics = dict()

    """
    Compile Features
    """
    def get_template_statistics(self):
        return self.template_statistics

    def calculate_template_statistics(self):
        self.template_statistics.update(self.calculate_p_wave_statistics())
        self.template_statistics.update(self.calculate_q_wave_statistics())
        self.template_statistics.update(self.calculate_t_wave_statistics())
        self.template_statistics.update(self.calculate_s_wave_statistics())

    def calculate_qrs_bounds(self):

        # Empty lists of QRS start and end times
        qrs_starts_sp = []
        qrs_ends_sp = []

        # Loop through templates
        for template in range(self.templates.shape[1]):

            # Get zero crossings before the R-Peak
            pre_qrs_zero_crossings = np.where(
                np.diff(np.sign(self.templates[0:self.template_rpeak_sp, template]))
            )[0]

            # Check length
            if len(pre_qrs_zero_crossings) >= 2:

                # Append QRS starting index
                qrs_starts_sp = np.append(qrs_starts_sp, pre_qrs_zero_crossings[-2])

            if len(qrs_starts_sp) > 0:

                self.qrs_start_sp = int(np.median(qrs_starts_sp))
                self.qrs_start_ts = self.qrs_start_sp / self.fs

            else:
                self.qrs_start_sp = int(self.template_before_sp / 2.0)

            # Get zero crossings after the R-Peak
            post_qrs_zero_crossings = np.where(
                np.diff(np.sign(self.templates[self.template_rpeak_sp:-1, template]))
            )[0]

            # Check length
            if len(post_qrs_zero_crossings) >= 2:

                # Append QRS ending index
                qrs_ends_sp = np.append(qrs_ends_sp, post_qrs_zero_crossings[-2])

            if len(qrs_ends_sp) > 0:

                self.qrs_end_sp = int(self.template_before_sp + np.median(qrs_ends_sp))
                self.qrs_end_ts = self.qrs_end_sp / self.fs

            else:
                self.qrs_end_sp = int(self.template_before_sp + self.template_after_sp / 2.0)

    def preprocess_pqrst(self):

        # Get QRS start point
        qrs_start_sp = self.template_rpeak_sp - self.qrs_start_sp_manual

        # Get QRS end point
        qrs_end_sp = self.template_rpeak_sp + self.qrs_end_sp_manual

        # Get QR median template
        qr_median_template = self.median_template[qrs_start_sp:self.template_rpeak_sp]
        print(qr_median_template)

        # Get RS median template
        rs_median_template = self.median_template[self.template_rpeak_sp:qrs_end_sp]
        print(rs_median_template)

        # Get QR templates
        qr_templates = self.templates[qrs_start_sp:self.template_rpeak_sp, :]
        print(qr_templates)

        # Get RS templates
        rs_templates = self.templates[self.template_rpeak_sp:qrs_end_sp, :]
        print(rs_templates)

        """
        Q-Wave
        """
        # Get array of Q-wave times (sp)
        self.q_times_sp = np.array(
            [qrs_start_sp + np.argmin(qr_templates[:, col]) for col in range(qr_templates.shape[1])]
        )

        # Get array of Q-wave amplitudes
        self.q_amps = np.array(
            [self.templates[self.q_times_sp[col], col] for col in range(self.templates.shape[1])]
        )

        # Get array of Q-wave times (sp)
        self.q_time_sp = qrs_start_sp + np.argmin(qr_median_template)

        # Get array of Q-wave amplitudes
        self.q_amp = self.median_template[self.q_time_sp]

        """
        P-Wave
        """
        # Get array of Q-wave times (sp)
        self.p_times_sp = np.array([
            np.argmax(self.templates[0:self.q_times_sp[col], col])
            for col in range(self.templates.shape[1])
        ])

        # Get array of Q-wave amplitudes
        self.p_amps = np.array(
            [self.templates[self.p_times_sp[col], col] for col in range(self.templates.shape[1])]
        )

        # Get array of Q-wave times (sp)
        self.p_time_sp = np.argmax(self.median_template[0:self.q_time_sp])

        # Get array of Q-wave amplitudes
        self.p_amp = self.median_template[self.p_time_sp]

        """
        S-Wave
        """
        # Get array of Q-wave times (sp)
        self.s_times_sp = np.array([
            self.template_rpeak_sp + np.argmin(rs_templates[:, col])
            for col in range(rs_templates.shape[1])
        ])

        # Get array of Q-wave amplitudes
        self.s_amps = np.array(
            [self.templates[self.s_times_sp[col], col] for col in range(self.templates.shape[1])]
        )

        # Get array of Q-wave times (sp)
        self.s_time_sp = self.template_rpeak_sp + np.argmin(rs_median_template)

        # Get array of Q-wave amplitudes
        self.s_amp = self.median_template[self.s_time_sp]

        """
        T-Wave
        """
        # Get array of Q-wave times (sp)
        self.t_times_sp = np.array([
            self.s_times_sp[col] + np.argmax(self.templates[self.s_times_sp[col]:, col])
            for col in range(self.templates.shape[1])
        ])

        # Get array of Q-wave amplitudes
        self.t_amps = np.array(
            [self.templates[self.t_times_sp[col], col] for col in range(self.templates.shape[1])]
        )

        # Get array of Q-wave times (sp)
        self.t_time_sp = self.s_time_sp + np.argmax(self.median_template[self.s_time_sp:])

        # Get array of Q-wave amplitudes
        self.t_amp = self.median_template[self.t_time_sp]

    def calculate_p_wave_statistics(self):

        # Empty dictionary
        p_wave_statistics = dict()

        # Get P-Wave energy bounds
        p_eng_start = self.p_time_sp - 10
        if p_eng_start < 0:
            p_eng_start = 0
        p_eng_end = self.p_time_sp + 10

        # Get end points
        start_sp = self.template_rpeak_sp - self.qrs_start_sp_manual

        # Calculate p-wave statistics
        if self.templates.shape[1] > 0:
            p_wave_statistics['p_wave_time'] = self.p_time_sp * 1 / self.fs
            p_wave_statistics['p_wave_time_std'] = np.std(self.p_times_sp * 1 / self.fs, ddof=1) * 1 / self.fs
            p_wave_statistics['p_wave_amp'] = self.p_amp
            p_wave_statistics['p_wave_amp_std'] = np.std(self.p_amps, ddof=1)
            p_wave_statistics['p_wave_eng'] = np.sum(np.power(self.median_template[p_eng_start:p_eng_end], 2))
        else:
            p_wave_statistics['p_wave_time'] = np.nan
            p_wave_statistics['p_wave_time_std'] = np.nan
            p_wave_statistics['p_wave_amp'] = np.nan
            p_wave_statistics['p_wave_amp_std'] = np.nan
            p_wave_statistics['p_wave_eng'] = np.nan

        """
        Calculate non-linear statistics
        """
        approximate_entropy = [
            pyeeg.ap_entropy(self.templates[0:start_sp, col], M=2, R=0.1*np.std(self.templates[0:start_sp, col]))
            for col in range(self.templates.shape[1])
        ]
        p_wave_statistics['p_wave_approximate_entropy_median'] = np.median(approximate_entropy)
        p_wave_statistics['p_wave_approximate_entropy_std'] = np.std(approximate_entropy, ddof=1)

        sample_entropy = [
            self.safe_check(
                ent.sample_entropy(
                    self.templates[0:start_sp, col],
                    sample_length=2,
                    tolerance=0.1*np.std(self.templates[0:start_sp, col])
                )[0]
            )
            for col in range(self.templates.shape[1])
        ]
        p_wave_statistics['p_wave_sample_entropy_median'] = np.median(sample_entropy)
        p_wave_statistics['p_wave_sample_entropy_std'] = np.std(sample_entropy, ddof=1)

        multiscale_entropy = [
            self.safe_check(
                ent.multiscale_entropy(
                    self.templates[0:start_sp, col],
                    sample_length=2,
                    tolerance=0.1*np.std(self.templates[0:start_sp, col])
                )[0]
            )
            for col in range(self.templates.shape[1])
        ]
        p_wave_statistics['p_wave_multiscale_entropy_median'] = np.median(multiscale_entropy)
        p_wave_statistics['p_wave_multiscale_entropy_std'] = np.std(multiscale_entropy, ddof=1)

        permutation_entropy = [
            self.safe_check(
                ent.permutation_entropy(
                    self.templates[0:start_sp, col],
                    m=2, delay=1
                )
            )
            for col in range(self.templates.shape[1])
        ]
        p_wave_statistics['p_wave_permutation_entropy_median'] = np.median(permutation_entropy)
        p_wave_statistics['p_wave_permutation_entropy_std'] = np.std(permutation_entropy, ddof=1)

        multiscale_permutation_entropy = [
            self.safe_check(
                ent.multiscale_permutation_entropy(
                    self.templates[0:start_sp, col],
                    m=2, delay=1, scale=1
                )[0]
            )
            for col in range(self.templates.shape[1])
        ]
        p_wave_statistics['p_wave_multiscale_permutation_entropy_median'] = np.median(multiscale_permutation_entropy)
        p_wave_statistics['p_wave_multiscale_permutation_entropy_std'] = np.std(multiscale_permutation_entropy, ddof=1)

        fisher_information = [
            fisher_info(self.templates[0:start_sp, col], tau=1, de=2)
            for col in range(self.templates.shape[1])
        ]
        p_wave_statistics['p_wave_fisher_info_median'] = np.median(fisher_information)
        p_wave_statistics['p_wave_fisher_info_std'] = np.std(fisher_information, ddof=1)

        higuchi_fractal = [
            hfd(self.templates[0:start_sp, col], k_max=10) for col in range(self.templates.shape[1])
        ]
        p_wave_statistics['p_wave_higuchi_fractal_median'] = np.median(higuchi_fractal)
        p_wave_statistics['p_wave_higuchi_fractal_std'] = np.std(higuchi_fractal, ddof=1)

        hurst_exponent = [
            pfd(self.templates[0:start_sp, col]) for col in range(self.templates.shape[1])
        ]
        p_wave_statistics['p_wave_hurst_exponent_median'] = np.median(hurst_exponent)
        p_wave_statistics['p_wave_hurst_exponent_std'] = np.std(hurst_exponent, ddof=1)

        svd_entr = [
            svd_entropy(self.templates[0:start_sp, col], tau=2, de=2)
            for col in range(self.templates.shape[1])
        ]
        p_wave_statistics['p_wave_svd_entropy_median'] = np.median(svd_entr)
        p_wave_statistics['p_wave_svd_entropy_std'] = np.std(svd_entr, ddof=1)

        return p_wave_statistics

    def calculate_q_wave_statistics(self):

        # Empty dictionary
        q_wave_statistics = dict()

        # Calculate p-wave statistics
        if self.templates.shape[1] > 0:
            q_wave_statistics['q_wave_time'] = self.q_time_sp * 1 / self.fs
            q_wave_statistics['q_wave_time_std'] = np.std(self.q_times_sp * 1 / self.fs, ddof=1) * 1 / self.fs
            q_wave_statistics['q_wave_amp'] = self.q_amp
            q_wave_statistics['q_wave_amp_std'] = np.std(self.q_amps, ddof=1)
        else:
            q_wave_statistics['q_wave_time'] = np.nan
            q_wave_statistics['q_wave_time_std'] = np.nan
            q_wave_statistics['q_wave_amp'] = np.nan
            q_wave_statistics['q_wave_amp_std'] = np.nan

        return q_wave_statistics

    def calculate_s_wave_statistics(self):

        # Empty dictionary
        s_wave_statistics = dict()

        # Calculate p-wave statistics
        if self.templates.shape[1] > 0:
            s_wave_statistics['s_wave_time'] = self.s_time_sp * 1 / self.fs
            s_wave_statistics['s_wave_time_std'] = np.std(self.s_times_sp * 1 / self.fs, ddof=1) * 1 / self.fs
            s_wave_statistics['s_wave_amp'] = self.s_amp
            s_wave_statistics['s_wave_amp_std'] = np.std(self.s_amps, ddof=1)
        else:
            s_wave_statistics['s_wave_time'] = np.nan
            s_wave_statistics['s_wave_time_std'] = np.nan
            s_wave_statistics['s_wave_amp'] = np.nan
            s_wave_statistics['s_wave_amp_std'] = np.nan

        return s_wave_statistics

    def calculate_t_wave_statistics(self):

        # Empty dictionary
        t_wave_statistics = dict()

        # Get T-Wave energy bounds
        t_eng_start = self.t_time_sp - 10
        t_eng_end = self.t_time_sp + 10
        if t_eng_end > self.templates.shape[0] - 1:
            t_eng_end = self.templates.shape[0] - 1

        # Get end points
        end_sp = self.template_rpeak_sp + self.qrs_end_sp_manual

        # Calculate p-wave statistics
        if self.templates.shape[1] > 0:
            t_wave_statistics['t_wave_time'] = self.t_time_sp * 1 / self.fs
            t_wave_statistics['t_wave_time_std'] = np.std(self.t_times_sp * 1 / self.fs, ddof=1) * 1 / self.fs
            t_wave_statistics['t_wave_amp'] = self.t_amp
            t_wave_statistics['t_wave_amp_std'] = np.std(self.t_amps, ddof=1)
            t_wave_statistics['t_wave_eng'] = np.sum(np.power(self.median_template[t_eng_start:t_eng_end], 2))
        else:
            t_wave_statistics['t_wave_time'] = np.nan
            t_wave_statistics['t_wave_time_std'] = np.nan
            t_wave_statistics['t_wave_amp'] = np.nan
            t_wave_statistics['t_wave_amp_std'] = np.nan
            t_wave_statistics['t_wave_eng'] = np.nan

        """
        Calculate non-linear statistics
        """
        approximate_entropy = [
            pyeeg.ap_entropy(self.templates[end_sp:, col], M=2, R=0.1*np.std(self.templates[end_sp:, col]))
            for col in range(self.templates.shape[1])
        ]
        t_wave_statistics['t_wave_approximate_entropy_median'] = np.median(approximate_entropy)
        t_wave_statistics['t_wave_approximate_entropy_std'] = np.std(approximate_entropy, ddof=1)

        sample_entropy = [
            self.safe_check(
                ent.sample_entropy(
                    self.templates[end_sp:, col],
                    sample_length=2,
                    tolerance=0.1*np.std(self.templates[end_sp:, col])
                )[0]
            )
            for col in range(self.templates.shape[1])
        ]
        t_wave_statistics['t_wave_sample_entropy_median'] = np.median(sample_entropy)
        t_wave_statistics['t_wave_sample_entropy_std'] = np.std(sample_entropy, ddof=1)

        multiscale_entropy = [
            self.safe_check(
                ent.multiscale_entropy(
                    self.templates[end_sp:, col],
                    sample_length=2,
                    tolerance=0.1*np.std(self.templates[end_sp:, col])
                )[0]
            )
            for col in range(self.templates.shape[1])
        ]
        t_wave_statistics['t_wave_multiscale_entropy_median'] = np.median(multiscale_entropy)
        t_wave_statistics['t_wave_multiscale_entropy_std'] = np.std(multiscale_entropy, ddof=1)

        permutation_entropy = [
            self.safe_check(
                ent.permutation_entropy(
                    self.templates[end_sp:, col],
                    m=2, delay=1
                )
            )
            for col in range(self.templates.shape[1])
        ]
        t_wave_statistics['t_wave_permutation_entropy_median'] = np.median(permutation_entropy)
        t_wave_statistics['t_wave_permutation_entropy_std'] = np.std(permutation_entropy, ddof=1)

        multiscale_permutation_entropy = [
            self.safe_check(
                ent.multiscale_permutation_entropy(
                    self.templates[end_sp:, col],
                    m=2, delay=1, scale=1
                )[0]
            )
            for col in range(self.templates.shape[1])
        ]
        t_wave_statistics['t_wave_multiscale_permutation_entropy_median'] = np.median(multiscale_permutation_entropy)
        t_wave_statistics['t_wave_multiscale_permutation_entropy_std'] = np.std(multiscale_permutation_entropy, ddof=1)

        return t_wave_statistics