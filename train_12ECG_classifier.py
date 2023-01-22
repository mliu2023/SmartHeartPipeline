#!/usr/bin/env python

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import sklearn.model_selection
import os
from tqdm import tqdm
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
    filenames = [] # list of filepaths 
    val_filenames = [] #list of filenames
    labels = [] # list of one-hot encoded tensors, each of length 27
    label_filenames = [] # list of filepaths to header files
    output_filenames = [] # list of filepaths to output files
    for file in os.listdir(input_directory):
        filepath = os.path.join(input_directory, file)
        if not file.lower().startswith('.') and file.lower().endswith('mat') and os.path.isfile(filepath):
            heafile = filepath.replace('mat', 'hea')
            if np.sum(get_labels_from_header(heafile, classes)) != 0:
                filenames.append(filepath)
                val_filenames.append(file)
        if not file.lower().startswith('.') and file.lower().endswith('hea') and os.path.isfile(filepath):
            if np.sum(get_labels_from_header(filepath, classes)) != 0:
                labels.append(get_labels_from_header(filepath, classes))
                label_filenames.append(filepath)
                output_filenames.append(os.path.join(output_directory, file.replace('.hea', '.csv')))
    print(len(filenames))
    print(len(labels))
    filenames = np.array(filenames)
    val_filenames = np.array(val_filenames)
    labels = np.array(labels)
    label_filenames = np.array(label_filenames)
    output_filenames = np.array(output_filenames)

    # define model, training parameters, loss, optimizer, and learning rate scheduler
    config = __import__(config_file)

    total_epochs = config.total_epochs
    min_length = config.min_length
    window_stride = config.window_stride
    window_size = config.window_size

    #### Weighted Loss ####
    loss = config.loss
    dx_mapping_df = pd.read_csv('dx_mapping_scored.csv')
    pos = torch.tensor(dx_mapping_df['Total'])
    print(pos)
    print(torch.sum(pos))
    pos_weight = (torch.sum(pos)-pos)/pos
    pos_weight = pos_weight.cuda()
    print(pos_weight)
    loss.pos_weight = pos_weight


    # composing preprocessing methods
    train_transforms = Compose([Resample(), Normalize(), NotchFilter(), ZeroPadding(min_length=min_length), RandomCropping(crop_size=config.window_size)])
    val_transforms = Compose([Resample(), Normalize(), NotchFilter(), ZeroPadding(min_length=min_length)])

    model_list = []
    thresholds_list = []
    print(filenames)
    # train model using 10-fold cross validation
    k_fold = sklearn.model_selection.KFold(n_splits=10, shuffle=True, random_state=0)
    for i, (train_indices, val_indices) in enumerate(k_fold.split(filenames, labels)):
        # initializing the datasets
        train_set = ECGDataset(filenames=filenames[train_indices], labels=labels[train_indices], transforms=train_transforms)
        val_set = ECGDataset(filenames=filenames[val_indices], labels=labels[val_indices], transforms=val_transforms)
        
        # creating the data loaders (generators)
        train_loader = DataLoader(dataset=train_set, batch_size=8, shuffle=True)
        val_loader = DataLoader(dataset=val_set, batch_size=1, shuffle=False)
        
        # imports from config file
        model = config.model
        optimizer = config.optimizer
        lr_scheduler = config.lr_scheduler
        # training + validation loop
        print(f'Training on fold {i+1}')
        for epoch in range(1, total_epochs+1):
            print(f'Epoch {epoch}/{total_epochs}')
            train(model, train_loader, loss, optimizer, lr_scheduler)
            val(model, val_loader, classes, val_filenames[val_indices], window_size, window_stride, output_directory)
            run(label_filenames[val_indices], output_filenames[val_indices], 'scores.csv')

        # overall threshold training
        best_score, best_threshold = 0, 0.5
        for threshold in np.linspace(0.1, 0.9, 9):
            _, _, _, _, _, _, _, _, _, _, score = evaluate_score(label_filenames[val_indices], output_filenames[val_indices], threshold, verbose=False)
            if(score > best_score):
                best_threshold = threshold
                best_score = score

        # class threshold training
        model_threshold = np.zeros(len(classes))
        for i in range(len(classes)):
            best_class_score, best_class_threshold = 0, 0.5
            for class_threshold in np.linspace(0.1, 0.9, 9):
                thresholds = np.full(len(classes), best_threshold)
                thresholds[i] = class_threshold
                _, _, _, _, _, _, _, _, _, _, class_score = evaluate_score(label_filenames[val_indices], output_filenames[val_indices], thresholds, verbose=False)
                if(class_score > best_class_score):
                    best_class_threshold = class_threshold
                    best_class_score = class_score
            model_threshold[i] = best_class_threshold
        
        # saving model and threshold
        thresholds_list.append(model_threshold)
        print(model_threshold)
        model_list.append(model)
        break

# performs one training iteration
def train(model, train_loader, loss, optimizer, lr_scheduler):
    with tqdm(train_loader, unit='batch') as minibatch:
        for (x, demographics), y in minibatch:
            # .cuda() loads the data onto the GPU
            # this code will not run without a compatible NVIDIA GPU
            x = x.cuda()
            demographics = demographics.cuda()
            y = y.float().cuda()

            optimizer.zero_grad()
            y_pred = model(x, demographics)
            train_loss = loss(y_pred, y)

            train_loss.backward()
            optimizer.step()

            minibatch.set_postfix(loss=train_loss.item())
    if lr_scheduler is not None:
        lr_scheduler.step()


# performs one validation iteration and outputs the challenge score
def val(model, val_loader, classes, filenames, window_size, window_stride, output_directory, thresholds = None):
    with tqdm(val_loader, unit='batch') as minibatch:
        for batch_num, ((x, demographics), y) in enumerate(minibatch):
            # .cuda() loads the data onto the GPU
            # this code will not run without a compatible NVIDIA GPU
            demographics = demographics.cuda()
            y = y.cuda()

            windows = []
            for i in range(0, x.shape[2]-window_size+1, window_stride):
                window = x[:, :, i: i+window_size]
                windows.append(window)
                if len(windows) > 10:
                    break
            windows = torch.vstack(windows).cuda()
            demographics = demographics.expand(windows.shape[0], demographics.shape[1])
            y_preds = model(windows, demographics)
            y_preds = torch.sigmoid(y_preds)
            scores = torch.mean(y_preds, 0)

            if thresholds is None:
                y_label_pred = scores > 0.5 # the threshold will be adjusted after the model is done training
            else:
                y_label_pred = scores > thresholds

            scores = scores.detach().cpu().numpy()
            y_label_pred = np.array(y_label_pred.detach().cpu().numpy(), dtype=np.uint8)

            save_challenge_predictions(output_directory,filenames[batch_num],scores=scores,labels=y_label_pred,classes=classes)

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

def evaluate_score(label_filenames, output_filenames, thresholds = None, verbose = True):

    verboseprint = print if verbose else lambda *a, **k: None

    # Define the weights, the SNOMED CT code for the normal class, and equivalent SNOMED CT codes.
    weights_file = 'weights.csv'
    normal_class = '426783006'
    equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]

    # Load the scored classes and the weights for the Challenge metric.
    verboseprint('Loading weights...')
    classes, weights = load_weights(weights_file, equivalent_classes)

    # Load the label and output files.
    verboseprint('Loading label and output files...')
    label_files, output_files = label_filenames, output_filenames
    labels = load_labels(label_files, classes, equivalent_classes)
    binary_outputs, scalar_outputs = load_outputs(output_files, classes, equivalent_classes)
    if(thresholds is not None):
        binary_outputs = scalar_outputs > thresholds

    # Evaluate the model by comparing the labels and outputs.
    verboseprint('Evaluating model...')

    verboseprint('- AUROC and AUPRC...')
    auroc, auprc, auroc_classes, auprc_classes = compute_auc(labels, scalar_outputs)

    verboseprint('- Accuracy...')
    accuracy = compute_accuracy(labels, binary_outputs)

    verboseprint('- F-measure...')
    f_measure, f_measure_classes = compute_f_measure(labels, binary_outputs)

    verboseprint('- F-beta and G-beta measures...')
    f_beta_measure, g_beta_measure = compute_beta_measures(labels, binary_outputs, beta=2)

    verboseprint('- Challenge metric...')
    challenge_metric = compute_challenge_metric(weights, labels, binary_outputs, classes, normal_class)

    verboseprint('Done.')

    # Return the results.
    return classes, auroc, auprc, auroc_classes, auprc_classes, accuracy, f_measure, f_measure_classes, f_beta_measure, g_beta_measure, challenge_metric

def run(label_filenames, output_filenames, challenge_score_file):
    classes, auroc, auprc, auroc_classes, auprc_classes, accuracy, f_measure, f_measure_classes, f_beta_measure, g_beta_measure, challenge_metric = evaluate_score(label_filenames, output_filenames, verbose=False)
    output_string = 'AUROC,AUPRC,Accuracy,F-measure,Fbeta-measure,Gbeta-measure,Challenge metric\n{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}'.format(auroc, auprc, accuracy, f_measure, f_beta_measure, g_beta_measure, challenge_metric)
    with open(challenge_score_file, 'w') as f:
        f.write(output_string)
