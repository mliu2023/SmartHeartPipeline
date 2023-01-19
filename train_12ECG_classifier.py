#!/usr/bin/env python

import torch
from torch.utils.data import DataLoader
import numpy as np
import sklearn.model_selection
import os
from tqdm import tqdm
import importlib
from dataset import ECGDataset
from evaluate_12ECG_score import *
from driver import *
from transform_utils import *

def train_12ECG_classifier(input_directory, output_directory, config_file):

    # initializing class labels
    classes = [270492004, 164889003, 164890007, 426627000, 713427006, 713426002, 445118002, 39732003, 164909002, 
               251146004, 698252002, 10370003, 284470004, 427172004, 164947007, 111975006, 164917005, 47665007, 
               59118001, 427393009, 426177001, 426783006, 427084000, 63593006, 164934002, 59931005, 17338001]
    for i in range(len(classes)):
        classes[i] = str(classes[i])

    # retrieving filenames and labels from the data folder
    filenames = [] # list of filenames
    names = []
    labels = [] # list of one-hot encoded tensors, each of length 27
    for file in os.listdir(input_directory):
        filepath = os.path.join(input_directory, file)
        if not file.lower().startswith('.') and file.lower().endswith('mat') and os.path.isfile(filepath):
            filenames.append(filepath)
            names.append(file)
        if not file.lower().startswith('.') and file.lower().endswith('hea') and os.path.isfile(filepath):
            labels.append(get_labels_from_header(filepath, classes))
    filenames = np.array(filenames)
    names = np.array(names)
    labels = np.array(labels)
    # define model, training parameters, loss, optimizer, and learning rate scheduler
    config = __import__(config_file)
    model = config.model
    warmup_epochs = config.warmup_epochs
    total_epochs = config.total_epochs
    loss = config.loss
    max_length = config.max_length
    optimizer = config.optimizer
    lr_scheduler = config.lr_scheduler
    stride = config.stride
    window_size = config.window_size
    # composing preprocessing methods
    train_transforms = Compose([Resample(), Normalize(), NotchFilter(), ZeroPadding(max_length=max_length), RandomCropping(crop_size=config.window_size)])
    val_transforms = Compose([Resample(), Normalize(), NotchFilter(), ZeroPadding(max_length=max_length)])

    # train model
    k_fold = sklearn.model_selection.KFold(n_splits=10)
    for i, (train_indices, val_indices) in enumerate(k_fold.split(filenames, labels)):
        # initializing the datasets
        train_set = ECGDataset(filenames=filenames[train_indices], labels=labels[train_indices], transforms=train_transforms)
        val_set = ECGDataset(filenames=filenames[val_indices], labels=labels[val_indices], transforms=val_transforms)

        # creating the data loaders (generators)
        train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True)
        val_loader = DataLoader(dataset=val_set, batch_size=1, shuffle=False)

        
        # training + validation loop
        print(f'Training on fold {i+1}')
        for epoch in range(1, total_epochs+1):
            print(f'Epoch {epoch}/{total_epochs}')
            #train(model, train_loader, loss, optimizer)
            val(model, val_loader, classes, names[val_indices], window_size, stride, output_directory)
            if lr_scheduler is not None:
                lr_scheduler.step(epoch)
# performs one training iteration
def train(model, train_loader, loss, optimizer):
    with tqdm(train_loader, unit='batch') as minibatch:
        for (x, demographics), y in minibatch:
            # .cuda() loads the data onto the GPU
            # this code will not run without a compatible NVIDIA GPU
            x = x.cuda()
            demographics = demographics.cuda()
            y = y.float().cuda()

            y_pred = model(x, demographics)

            train_loss = loss(y_pred, y)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            minibatch.set_postfix(loss=train_loss.item())


# performs one validation iteration and outputs the challenge score
def val(model, val_loader, classes, filenames, window_size, stride, output_directory, thresholds = None):
    with tqdm(val_loader, unit='batch') as minibatch:
        for batch_num, ((x, demographics), y) in enumerate(minibatch):
            # .cuda() loads the data onto the GPU
            # this code will not run without a compatible NVIDIA GPU
            #x = x.cuda()
            demographics = demographics.cuda()
            y = y.cuda()
            num_windows = 0
            windows = []
            scores = torch.zeros((x.shape[0], 27,), dtype = torch.float).cuda()
            for i in range(0,min(7500,x.shape[2])-window_size+1, stride):
                window = x[:,:,i:i+window_size]
                windows.append(window.unsqueeze(0))
                num_windows += 1
            windows = torch.squeeze(torch.vstack(windows)).cuda()
            demographics = demographics.expand(windows.shape[0], demographics.shape[1])
            preds = model(windows, demographics)
            preds = torch.sigmoid(preds)
            scores = torch.mean(preds, 0)

            y_label_pred = scores > 0.5 # the threshold will be adjusted after the model is done training
            scores = scores.detach().cpu().numpy()
            
            y_label_pred = np.array(y_label_pred.detach().cpu().numpy(), dtype=np.uint8)

            save_challenge_predictions(output_directory,filenames[batch_num],scores=scores,labels=y_label_pred,classes=classes)

# def save_predictions(output_directory,filenames,scores,labels,classes):
#     for filename in filenames:
#         recording = os.path.splitext(filename)[0]
#         new_file = filename.replace('.mat','.csv')
#         output_file = os.path.join(output_directory,new_file)

#         # Include the filename as the recording number
#         recording_string = '#{}'.format(recording)
#         class_string = ','.join(classes)
#         label_string = ','.join(str(i) for i in labels)
#         score_string = ','.join(str(i) for i in scores)

#         with open(output_file, 'w') as f:
#             f.write(recording_string + '\n' + class_string + '\n' + label_string + '\n' + score_string + '\n')

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
