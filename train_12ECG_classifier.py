#!/usr/bin/env python

import torch
from torch.utils.data import DataLoader
import numpy as np
import sklearn
import os
import tqdm
from dataset import ECGDataset
from transform_utils import *

def train_12ECG_classifier(input_directory, output_directory):

    # initializing class labels
    classes = [270492004, 164889003, 164890007, 426627000, 713427006, 713426002, 445118002, 39732003, 164909002, 
               251146004, 698252002, 10370003, 284470004, 427172004, 164947007, 111975006, 164917005, 47665007, 
               59118001, 427393009, 426177001, 426783006, 427084000, 63593006, 164934002, 59931005, 17338001]
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
    filenames = np.array(filenames)
    labels = np.array(labels)

    # composing preprocessing methods
    train_transforms = Compose([Resample, Normalize, Notch_filter])
    val_transforms = Compose([Resample, Normalize, Notch_filter])

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

        # define training parameters, loss, optimizer, and learning rate scheduler
        warmup_epochs = 10
        total_epochs = 100
        lr = 1e-3
        loss = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epochs-warmup_epochs)
        lr_scheduler = Warmup_LR(optimizer=optimizer,
                                warmup_iteration=warmup_epochs,
                                target_lr=lr,
                                after_scheduler=cosine_scheduler)

        # training + validation loop
        print(f'Training on fold {i+1}')
        for epoch in range(1, total_epochs+1):
            print(f'Epoch {epoch}/{total_epochs}')
            train(model, train_loader, loss, optimizer, lr_scheduler)
            val(model, val_loader, classes, filenames[val_indices], output_directory)

# performs one training iteration
def train(model, train_loader, loss, optimizer, lr_scheduler=None):
    with tqdm(train_loader, unit='batch') as minibatch:
        for x,y in minibatch:
            # .cuda() loads the data onto the GPU
            # this code will not run without a compatible NVIDIA GPU
            x = x.cuda()
            y = y.cuda()

            y_pred = model(x)
            loss = loss(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            minibatch.set_postfix(loss=loss.item())
    if(lr_scheduler is not None):
        lr_scheduler.step()

# performs one validation iteration and outputs the challenge score
def val(model, val_loader, classes, filenames, output_directory):
    with tqdm(val_loader, unit='batch') as minibatch:
        for x,y in minibatch:
            # .cuda() loads the data onto the GPU
            # this code will not run without a compatible NVIDIA GPU
            x = x.cuda()
            y = y.cuda()

            y_pred = model(x)

            # evaluating the challenge score
            y_pred = torch.sigmoid(y_pred).squeeze() # sigmoid activation for multilabel classification
            y_label_pred = y_pred > 0.5 # the threshold will be adjusted after the model is done training

            y_pred = y_pred.detach().cpu().numpy()
            y_label_pred = np.array(y_label_pred.detach().cpu().numpy(), dtype=np.uint8)


            save_predictions(output_directory,filenames,scores=y_pred,labels=y_label_pred,classes=classes)

def save_predictions(output_directory,filenames,scores,labels,classes):
    for filename in filenames:
        recording = os.path.splitext(filename)[0]
        new_file = filename.replace('.mat','.csv')
        output_file = os.path.join(output_directory,new_file)

        # Include the filename as the recording number
        recording_string = '#{}'.format(recording)
        class_string = ','.join(classes)
        label_string = ','.join(str(i) for i in labels)
        score_string = ','.join(str(i) for i in scores)

        with open(output_file, 'w') as f:
            f.write(recording_string + '\n' + class_string + '\n' + label_string + '\n' + score_string + '\n')

# gets the class labels from the given header file(s)
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

# warmup learning rate scheduler
class Warmup_LR(object):
    def __init__(self, optimizer, warmup_iteration, target_lr, after_scheduler=None):
        self.optimizer = optimizer
        self.warmup_iteration = warmup_iteration
        self.target_lr = target_lr
        self.after_scheduler = after_scheduler
        self.step(1)

    def warmup_learning_rate(self, current_iteration):
        warmup_lr = self.target_lr*float(current_iteration)/float(self.warmup_iteration)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = warmup_lr

    def step(self, current_iteration):
        if current_iteration <= self.warmup_iteration:
            self.warmup_learning_rate(current_iteration)
        else:
            self.after_scheduler.step(current_iteration-self.warmup_iteration)
    
    def load_state_dict(self, state_dict):
        self.after_scheduler.load_state_dict(state_dict)
