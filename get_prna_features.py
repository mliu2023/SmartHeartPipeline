# 1 HRmin
# 2 T wave multiscale permutation entropy σ
# 3 HRmax
# 4 T wave multiscale permutation entropy median
# 5 RMSSD
# 6 P wave correlation coefficient
# 7 RR interval median
# 8 Heart rate μ
# 9 RR interval intra-cluster distance, cluster 3
# 10 RR interval Fisher information
# 11 pNN60
# 12 SWT decomposition level 4 entropy
# 13 RR interval intra-cluster distance, cluster 2
# 14 Heart rate activity
# 15 ∆ RRmin
# 16 T wave permutation entropy σ
# 17 P wave sample entropy σ
# 18 SWT decomposition level 3 entropy
# 19 Median p wave approximate entropy
# 20 R peak approximate entropy
import numpy as np
from scipy.signal import find_peaks

def extract_features(ecg_signal):
    # assuming 500hz sample freq
    sample_frequency = 500
    # 1. HRmin
    r_peaks, _ = find_peaks(ecg_signal)
    rr_intervals = np.diff(r_peaks)
    hr_min = np.min(sample_frequency/rr_intervals)
    
    # 3. HRmax
    hr_max = np.max(sample_frequency/rr_intervals)
    
    # 8. Heart rate μ
    hr_mean = np.mean(sample_frequency/rr_intervals)