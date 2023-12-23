#!/usr/bin/env python

from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os, json
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from dataset import ECGDataset
from utils.transform_utils import *

def ECG_visualization(input_directory, output_directory):

    # initializing class labels
    classes = [270492004, 164889003, 164890007, 426627000, 713427006, 713426002, 445118002, 39732003, 164909002, 
               251146004, 698252002, 10370003, 284470004, 427172004, 164947007, 111975006, 164917005, 47665007, 
               59118001, 427393009, 426177001, 426783006, 427084000, 63593006, 164934002, 59931005, 17338001]

    for i in range(len(classes)):
        classes[i] = str(classes[i])

    #### retrieving filenames and labels from the data folder ####
    # .mat files have the ecg data, while .hea files are the header files and .csv files are for outputs
    filepaths = [] # list of filepaths \data\A0001.mat
    val_filenames = [] # list of filenames \A0001.mat
    labels = [] # list of one-hot encoded tensors, each of length 27
    label_filepaths = [] # list of filepaths to header files \data\A0001.hea
    for file in os.listdir(input_directory):
        filepath = os.path.join(input_directory, file)
        if not file.lower().startswith('.') and file.lower().endswith('mat') and os.path.isfile(filepath):
            heafile = filepath.replace('mat', 'hea')
            if np.sum(get_labels_from_header(heafile, classes)) != 0:
                filepaths.append(filepath)
                val_filenames.append(file)
        if not file.lower().startswith('.') and file.lower().endswith('hea') and os.path.isfile(filepath):
            if np.sum(get_labels_from_header(filepath, classes)) != 0:
                labels.append(get_labels_from_header(filepath, classes))
                label_filepaths.append(filepath)
    print(f'Total number of files: {len(filepaths)}')
    filepaths = np.array(filepaths)
    val_filenames = np.array(val_filenames)
    labels = np.array(labels)
    label_filepaths = np.array(label_filepaths)

    features = pd.read_csv('features.csv').to_numpy()[:, 1:]
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    with open('rpeaks.json', 'r') as f:
        rpeaks = json.load(f)

    # composing preprocessing methods
    min_length, window_size = 7500, 7500
    wavelet, level = 'db3', 3
    transforms = Compose([Resample(), Normalize(), NotchFilter(), DWT(wavelet=wavelet, level=level), ZeroPadding(min_length=min_length), RandomCropping(crop_size=window_size)])

    # taking in indices to visualize
    indices = [int(index) for index in input('Select filename indices to visualize (leave just a space between indices): ').split()]
    filepaths = filepaths[indices]
    labels = labels[indices]
    features = features[indices]
    rpeaks = [rpeaks[i] for i in indices]
    dataset = ECGDataset(filepaths=filepaths, labels=labels, features=features, rpeaks=rpeaks, transforms=transforms)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    for batch_num, ((x, demographics), y, _), in enumerate(loader):
        x = x.squeeze()
        demographics = demographics.squeeze()
        y = y.squeeze()
        plt.figure(figsize=(20, 15))
        for i in range(0, 12):
            plt.subplot(3, 4, i+1)
            plt.plot(x[i][0:1000])
        plt.savefig(filepaths[batch_num].replace(input_directory, output_directory).rstrip('.mat') + f'_{wavelet}_level{level}_visualization.png')



# gets the class labels from the given header file(s)
def get_labels_from_header(header_file, classes):
    with open(header_file, 'r') as file:
        lines = file.readlines()
        labels = np.zeros(len(classes))
        line15 = lines[15] # Example line: #Dx: 164865005,164951009,39732003,426783006
        arr = line15.strip().split(' ')
        diagnoses_arr = arr[1].split(',')
        for diagnosis in diagnoses_arr:
            if diagnosis in classes:
                class_index = classes.index(diagnosis)
                labels[class_index] = 1
        return labels

if __name__ == '__main__':
    # Parse arguments.
    input_directory = 'data'
    output_directory = 'data_visualization'

    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    print('Running visualization code...')

    ECG_visualization(input_directory, output_directory)

    print('Done.')
