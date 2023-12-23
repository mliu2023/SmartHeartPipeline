#!/usr/bin/env python

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import sklearn.model_selection
import os
from tqdm import tqdm
from dataset import ECGDataset
from utils.evaluate_12ECG_score import *
from utils.driver import *
from utils.transform_utils import *
from utils.early_stopping import EarlyStopping
from models.ResNetAutoencoder import Autoencoder
from utils.warmup_lr import GradualWarmupScheduler

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

def train_autoencoder(input_directory):

    #### retrieving filenames and labels from the data folder ####
    filepaths = [] # list of filepaths \data\A0001.mat
    val_filenames = [] # list of filenames \A0001.mat
    for file in os.listdir(input_directory):
        filepath = os.path.join(input_directory, file)
        if not file.lower().startswith('.') and file.lower().endswith('mat') and os.path.isfile(filepath):
            filepaths.append(filepath)
            val_filenames.append(file)
    print(f'Total number of files: {len(filepaths)}')
    filepaths = np.array(filepaths)
    val_filenames = np.array(val_filenames)
    labels = np.zeros((len(filepaths), 27))
    print(f'Filepaths: {filepaths}')

    #### define model, training parameters, and loss functions ####
    total_epochs = 20
    warmup_epochs = 10
    min_length = 7500
    window_size = 7500
    window_stride = 3000
    loss = nn.MSELoss()
    val_loss = nn.MSELoss()
    batch_size = 8

    #### composing preprocessing methods ####
    train_transforms = Compose([Resample(), Normalize(), NotchFilter(), ZeroPadding(min_length=min_length), RandomCropping(crop_size=window_size)])
    val_transforms = Compose([Resample(), Normalize(), NotchFilter(), ZeroPadding(min_length=min_length)])

    #### comptuing hand-crafted features ####
    features = np.zeros((len(filepaths), 35))

    #### train model using 10-fold cross validation ####
    k_fold = sklearn.model_selection.KFold(n_splits=10, shuffle=True, random_state=0)
    for i, (train_indices, val_indices) in enumerate(k_fold.split(filepaths, labels)):
        # imports from config file
        model = Autoencoder()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        after_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epochs-warmup_epochs)
        lr_scheduler = GradualWarmupScheduler(optimizer, warmup_epochs, after_scheduler, value_flag=False)

        # initializing the datasets
        train_set = ECGDataset(filepaths=filepaths[train_indices], labels=labels[train_indices], features=features[train_indices], transforms=train_transforms)
        val_set = ECGDataset(filepaths=filepaths[val_indices], labels=labels[val_indices], features=features[val_indices], transforms=val_transforms)

        # creating the data loaders
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_set, batch_size=1, shuffle=False)

        # early stopping
        early_stopping_val = EarlyStopping(patience = 10, mode = 'min')
    
        # training + validation loop
        print(f'Training on fold {i+1}')
        for epoch in range(1, total_epochs+1):
            print(f'Epoch {epoch}/{total_epochs}')

            avg_train_loss = train(model, train_loader, loss, optimizer, lr_scheduler, device)

            avg_val_loss = val(model, val_loader, val_loss, window_size, window_stride)

            stop = early_stopping_val(avg_val_loss)
            print(f"Train loss: {avg_train_loss}")
            print(f"Val loss: {avg_val_loss}    Early stopping counter: {early_stopping_val.counter}")
            if(stop):
                break
        break


# performs one training iteration
def train(model, train_loader, loss, optimizer, lr_scheduler, device):
    total_train_loss = 0
    with tqdm(train_loader, unit='batch') as minibatch:
        for (x, _), _ in minibatch:
            x = x.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            train_loss = loss(y_pred, x)
            train_loss.backward()
            optimizer.step()
            total_train_loss += train_loss.item()
            minibatch.set_postfix(loss=train_loss.item())
    if lr_scheduler is not None:
        lr_scheduler.step()
    return total_train_loss/len(minibatch)


# performs one validation iteration and outputs the challenge score
def val(model, val_loader, loss, window_size, window_stride):
    total_val_loss = 0
    with tqdm(val_loader, unit='batch') as minibatch:
        for (x, _), _ in minibatch:
            x = x.to(device)

            windows = []
            for i in range(0, x.shape[2]-window_size+1, window_stride):
                window = x[:, :, i: i+window_size]
                windows.append(window)
                if len(windows) > 10:
                    break
            windows = torch.vstack(windows).to(device)

            y_pred = model(windows)
            val_loss = loss(y_pred, windows)
            total_val_loss += val_loss.item()
            minibatch.set_postfix(loss=val_loss.item())
    return total_val_loss/len(minibatch)

if __name__ == '__main__':
    # Parse arguments.
    input_directory = sys.argv[1]

    print('Running training code...')

    train_autoencoder(input_directory)

    print('Done.')