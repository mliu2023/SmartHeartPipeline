import torch
import numpy as np
import scipy.io as scio
from torch.utils.data import Dataset
from utils.feats.features import *
import pandas as pd

def get_frequency_from_header(header_file):
    with open(header_file, 'r') as file:
        first_line = file.readline()
        frequency = first_line.split(' ')[2]
        return int(frequency)

def get_demographics_from_header(header_file):
    with open(header_file, 'r') as file:
        lines = file.readlines()

        # retrieving age
        line13 = lines[13].strip().split(' ')
        if line13[1] == 'NaN':
            age = 50
        else:
            age = int(line13[1])
        age /= 100

        # retrieving gender
        gender = lines[14].strip().split(' ')[1]
        female = 0
        male = 0
        if gender == 'Female' or gender == 'female':
            female = 1
        elif gender == 'Male' or gender == 'male':
            male = 1
        return torch.tensor([age,female,male], dtype=torch.float)

def get_ecg_features(x, freq):
    return extract_features(x, freq)

class ECGDataset(Dataset):
    def __init__(self, filenames, labels, transforms=None):
        super(ECGDataset, self).__init__()
        self.filenames = filenames
        self.labels = labels
        self.transforms = transforms # transforms is a sequence of transformations that should be applied to the signal data
    def __len__(self):
        return len(self.filenames)
    def __getitem__(self, index):
        filename = self.filenames[index]
        headname = filename.replace('.mat', '.hea')
        x = scio.loadmat(filename)['val']
        x = np.array(x, dtype=np.float32)

        # apply the transforms; source frequency is needed to correctly resample
        x_transformed = self.transforms(ecg_signal=x, source_freq=get_frequency_from_header(headname))
        x_transformed = torch.tensor(x_transformed, dtype=torch.float)

        print(x_transformed.shape)
        # retrieving age and gender
        # demographics = get_demographics_from_header(headname)
        # retrieving hand-crafted features
        # for i in range(12):
        #     ecg_features = get_ecg_features(x[i], get_frequency_from_header(headname))
        #     print(ecg_features)

        feats = Features(x[1], get_frequency_from_header(headname), feature_groups=['heart_rate_variability_statistics', 'template_statistics'])
        feats.calculate_features([3,45])
        pd.DataFrame.to_csv(feats.get_features()[['heart_rate_min', 't_wave_multiscale_permutation_entropy_std', 'heart_rate_max', 't_wave_multiscale_permutation_entropy_median',  'diff_rri_rms', 'p_wave_corr_coeff_median', 'rri_median', 'heart_rate_mean', 'heart_rate_median', 'rri_cluster_ssd_3', 'pnn60', 'rri_cluster_ssd_2', 'rri_min', 't_wave_permutation_entropy_std', 'p_wave_approximate_entropy_median']], "heart_rate_features.csv", mode='a', header=False)
        # concatenating the additional features
    
        # additional_features = torch.concatenate((demographics, ecg_features), dim=1)

        # return ((x_transformed, demographics), self.labels[index])
