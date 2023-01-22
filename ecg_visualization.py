#!/usr/bin/env python

import torch
from torch.utils.data import DataLoader
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from dataset import ECGDataset
from transform_utils import *

def ECG_visualization(input_directory, output_directory):

    # initializing class labels
    classes = [270492004, 164889003, 164890007, 426627000, 713427006, 713426002, 445118002, 39732003, 164909002, 
               251146004, 698252002, 10370003, 284470004, 427172004, 164947007, 111975006, 164917005, 47665007, 
               59118001, 427393009, 426177001, 426783006, 427084000, 63593006, 164934002, 59931005, 17338001]

    for i in range(len(classes)):
        classes[i] = str(classes[i])

    # retrieving filenames and labels from the data folder
    filenames = [] # list of filepaths 
    labels = [] # list of one-hot encoded tensors, each of length 27
    label_filenames = [] # list of filepaths to header files
    for file in os.listdir(input_directory):
        filepath = os.path.join(input_directory, file)
        if not file.lower().startswith('.') and file.lower().endswith('mat') and os.path.isfile(filepath):
            heafile = filepath.replace('mat', 'hea')
            if np.sum(get_labels_from_header(heafile, classes)) != 0:
                filenames.append(filepath)
        if not file.lower().startswith('.') and file.lower().endswith('hea') and os.path.isfile(filepath):
            if np.sum(get_labels_from_header(filepath, classes)) != 0:
                labels.append(get_labels_from_header(filepath, classes))
                label_filenames.append(filepath)
    filenames = np.array(filenames)
    labels = np.array(labels)
    label_filenames = np.array(label_filenames)

    # composing preprocessing methods
    min_length, window_size = 1500, 1500
    transforms = Compose([Resample(), Normalize(), NotchFilter(), ZeroPadding(min_length=min_length), RandomCropping(crop_size=window_size)])

    # taking in indices to visualize
    indices = [int(index) for index in input('Select filename indices to visualize (leave just a space between indices): ').split()]
    filename_list = np.array(filenames[indices])
    label_list = np.array(labels[indices])
    dataset = ECGDataset(filenames=filename_list, labels=label_list, transforms=transforms)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    for batch_num, ((x, demographics), y), in enumerate(loader):
        x = x.squeeze()
        demographics = demographics.squeeze()
        y = y.squeeze()
        plt.figure(figsize=(20, 15))
        for i in range(0, 12):
            plt.subplot(3, 4, i+1)
            plt.plot(x[i])
        plt.savefig(filename_list[batch_num].replace(input_directory, output_directory).rstrip('.mat') + '_visualization.png')



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
