import torch
import numpy as np
import scipy.io as scio
from torch.utils.data import Dataset

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

def get_ecg_features(filename):
    pass

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

        # retrieving age and gender
        demographics = get_demographics_from_header(headname)

        # retrieving hand-crafted features
        # ecg_features = get_ecg_features(x)

        # concatenating the additional featuresS
        # additional_features = torch.concatenate((demographics, ecg_features), dim=1)

        return ((x_transformed, demographics), self.labels[index])
