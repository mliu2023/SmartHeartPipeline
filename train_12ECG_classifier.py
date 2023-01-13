#!/usr/bin/env python

import torch
from torch.utils.data import DataLoader
import numpy as np
import sklearn
import os
from get_12ECG_features import get_12ECG_features
from dataset import ECGDataset
from transform_utils import *

def train_12ECG_classifier(input_directory, output_directory):

    # initializing class labels
    classes = [270492004, 164889003, 164890007, 426627000, 713427006, 713426002, 445118002, 39732003, 164909002, 251146004, 698252002, 10370003, 284470004, 427172004, 164947007, 111975006, 164917005, 47665007, 59118001, 427393009, 426177001, 426783006, 427084000, 63593006, 164934002, 59931005, 17338001]
    for i in range(len(classes)):
        classes[i] = str(classes[i])

    # retrieving filenames and labels from the data folder
    filenames = [] # list of filenames
    labels = [] # list of one-hot encoded tensors, each of length 27
    for file in os.listdir(input_directory):
        filepath = os.path.join(input_directory, file)
        if not file.lower().startswith('.') and file.lower().endswith('mat') and os.path.isfile(filepath):
            filenames.append(filepath)
        if not file.lower().startswith('.') and file.lower().endswith('hea') and os.path.isfile(filepath):
            labels.append(get_labels_from_header(file))

    # composing preprocessing methods
    train_transforms = Compose([resample, normalize, notch_filter])
    val_transforms = Compose([resample, normalize, notch_filter])

    # train model
    stratified_k_fold = sklearn.model_selection.StratifiedKFold(n_splits=10)
    for i, (train_indices, val_indices) in enumerate(stratified_k_fold.split(filenames, labels)):
        # initializing the datasets
        train_set = ECGDataset(filenames=filenames[train_indices], labels=labels[train_indices], transforms=train_transforms)
        val_set = ECGDataset(filenames=filenames[val_indices], labels=labels[val_indices], transforms=val_transforms)

        # creating the data loaders (generators)
        train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True)
        val_loader = DataLoader(dataset=val_set, batch_size=32, shuffle=False)

        # define the model
        model = CoolResNet18ThatWeFoundOnTheInternet()

        # define training parameters and functions
        warmup_epochs = 10
        epochs = 100
        loss = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters, lr=1e-3, weight_decay=1e-5)
        lr_scheduler = Warmup_LR_Scheduler(optimizer)

        for i in range(epochs):
            print(f'Epoch {i+1}/{epochs}' + '\n')
            train(model, train_loader, loss, optimizer, lr_scheduler=None)
            val(model, val_loader)


def get_labels_from_header(header_file, classes):
    with open(header_file, 'r') as file:
        lines = file.readlines()
        labels = torch.zeros(len(classes))
        line15 = lines[15] # Example line: #Dx: 164865005,164951009,39732003,426783006
        arr = line15.strip().split(' ')
        diagnoses_arr = arr[1].split(',')
        for diagnosis in diagnoses_arr:
            if diagnosis in classes:
                class_index = classes.index(diagnosis)
                labels[class_index] = 1
        return labels
