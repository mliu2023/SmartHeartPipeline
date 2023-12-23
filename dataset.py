import torch
import numpy as np
import scipy.io as scio
from torch.utils.data import Dataset
import re

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
            age = int(re.findall("[0-9]+", lines[13])[0])
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


class ECGDataset(Dataset):
    def __init__(self, filepaths, labels, features, rpeaks=None, transforms=None):
        super(ECGDataset, self).__init__()
        self.filepaths = filepaths
        self.labels = labels
        self.features = features
        self.rpeaks = rpeaks
        self.transforms = transforms # transforms is a sequence of transformations that should be applied to the signal data

        if rpeaks is not None:
            for i in range(len(self.rpeaks)):
                for j in range(12):
                    if len(rpeaks[i][j]) > 20:
                        self.rpeaks[i][j] = self.rpeaks[i][j][0:20]
                    elif len(rpeaks[i][j]) < 20:
                        self.rpeaks[i][j].extend([-1 for _ in range(20-len(self.rpeaks[i][j]))])
            self.rpeaks = np.array(self.rpeaks)
    def __len__(self):
        return len(self.filepaths)
    def __getitem__(self, index):
        filepath = self.filepaths[index]
        headname = filepath.replace('.mat', '.hea')
        x = scio.loadmat(filepath)['val']
        x = np.array(x, dtype=np.float32)
        source_freq=get_frequency_from_header(headname)
        # apply the transforms; source frequency is needed to correctly resample
        x_transformed = self.transforms(ecg_signal=x, source_freq=source_freq)
        x_transformed = torch.tensor(x_transformed, dtype=torch.float)

        # retrieving age and gender
        demographics = get_demographics_from_header(headname)

        # retrieving additional hand-crafted features
        ecg_features = torch.tensor(self.features[index, [0, 1, 2, 4, 5, 11, 12, 13, 14, 15, 17, 18, 25, 31, 34]], dtype=torch.float)
        # ecg_features = torch.tensor(self.features[index, [0, 1, 2]], dtype=torch.float)

        # concatenating the additional features
        additional_features = torch.concat((demographics, ecg_features), dim=0)

        if self.rpeaks is not None:
            return ((x_transformed, additional_features), self.labels[index], self.rpeaks[index])
        else:
            return ((x_transformed, additional_features), self.labels[index], torch.tensor([0]))