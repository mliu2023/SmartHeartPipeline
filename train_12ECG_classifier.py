#!/usr/bin/env python

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import sklearn.model_selection
import os, json
import scipy.io as scio
from tqdm import tqdm
from dataset import ECGDataset
from utils.evaluate_12ECG_score import *
from utils.driver import *
from utils.transform_utils import *
from utils.histograms import *
from utils.early_stopping import EarlyStopping
from utils.get_prna_features import extract_features, extract_rpeaks
from dataset import get_frequency_from_header
from sklearn.preprocessing import MinMaxScaler


def train_12ECG_classifier(input_directory, output_directory, config_file):

    #### writer will output to the ./runs/ directory for tensorboard visualizations ####
    writer = SummaryWriter()

    #### initializing class labels ####
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
    output_filenames = [] # list of filepaths to output files \output\A0001.csv
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
                output_filenames.append(os.path.join(output_directory, file.replace('.hea', '.csv')))
    print(f'Total number of files: {len(filepaths)}')
    filepaths = np.array(filepaths)
    val_filenames = np.array(val_filenames)
    labels = np.array(labels)
    label_filepaths = np.array(label_filepaths)
    output_filenames = np.array(output_filenames)
    print(f'Filepaths: {filepaths}')
    print(f'Filenames: {val_filenames}')



    #### import parameters from config file ####
    sys.path.append('configs')
    config = __import__(config_file)
    device = config.device
    total_epochs = config.total_epochs
    min_length = config.min_length
    window_stride = config.window_stride
    window_size = config.window_size
    loss = config.loss
    val_loss = config.val_loss
    batch_size = config.batch_size

    projector = False # decide if we want a projector
    if hasattr(config, 'projector'):
        projector = config.projector
    use_rpeaks = False # decide if we want to pass rpeaks into model
    if hasattr(config, 'use_rpeaks'):
        use_rpeaks = config.use_rpeaks



    #### composing preprocessing methods ####
    # note to self: remove from here after run
    train_transforms = Compose([Resample(), Normalize(), NotchFilter(), ZeroPadding(min_length=min_length), RandomCropping(crop_size=window_size)])
    val_transforms = Compose([Resample(), Normalize(), NotchFilter(), ZeroPadding(min_length=min_length)])
    if use_rpeaks:
        train_transforms = Compose([Resample(), Normalize(), NotchFilter(), ZeroPadding(min_length=min_length), Cropping(crop_size=window_size)])
    feature_extraction_transforms = Compose([Resample(), Normalize(), NotchFilter()])



    #### comptuing hand-crafted features + rpeaks ####
    compute_features = False
    compute_rpeaks = False
    if(compute_features):
        print('Computing hand-crafted features: ')
        features = []
        for filepath in tqdm(filepaths):
            headerpath = filepath.replace('.mat', '.hea')
            x = scio.loadmat(filepath)['val']
            x = np.array(x, dtype=np.float32)
            x = feature_extraction_transforms(ecg_signal=x, source_freq=get_frequency_from_header(headerpath), rpeaks=None)
            feature_dict = extract_features(x[0])
            features.append(list(feature_dict.values()))
        features = np.vstack(features)
        feature_df = pd.DataFrame(features, index=val_filenames, columns=list(feature_dict.keys()))
        feature_df.to_csv('features.csv')
        exit(0)
    elif(compute_rpeaks):
        rpeaks = []
        for filepath in tqdm(filepaths):
            headerpath = filepath.replace('.mat', '.hea')
            x = scio.loadmat(filepath)['val']
            x = np.array(x, dtype=np.float32)
            x = feature_extraction_transforms(ecg_signal=x, source_freq=get_frequency_from_header(headerpath))
            rpeak_list = extract_rpeaks(x)
            rpeaks.append(rpeak_list)
        with open('rpeaks.json', 'w') as f:
            json.dump(rpeaks, f, indent=2)
        exit(0)
    else:
        features = pd.read_csv('features.csv').to_numpy()[:, 1:]
        scaler = MinMaxScaler()
        features = scaler.fit_transform(features)
        if use_rpeaks:
            with open('rpeaks.json', 'r') as f:
                rpeaks = json.load(f)



    #### train model using 10-fold cross validation ####
    model_list = []
    thresholds_list = []
    k_fold = sklearn.model_selection.KFold(n_splits=10, shuffle=True, random_state=0)
    for i, (train_indices, val_indices) in enumerate(k_fold.split(filepaths, labels)):
        # imports from config file
        model = config.model

        # calculate number of parameters
        parameters = 0
        for layer in model.parameters():
            if layer.requires_grad:
                parameters += layer.numel()
        print(f"Total Model Parameters: {parameters}")
        optimizer = config.optimizer
        lr_scheduler = config.lr_scheduler

        #### data loading ####
        if use_rpeaks:
            train_set = ECGDataset(filepaths=filepaths[train_indices], labels=labels[train_indices], features=features[train_indices], rpeaks=[rpeaks[j] for j in train_indices], transforms=train_transforms)
            val_set = ECGDataset(filepaths=filepaths[val_indices], labels=labels[val_indices], features=features[val_indices], rpeaks=[rpeaks[j] for j in train_indices], transforms=val_transforms)
        else:
            train_set = ECGDataset(filepaths=filepaths[train_indices], labels=labels[train_indices], features=features[train_indices], rpeaks=None, transforms=train_transforms)
            val_set = ECGDataset(filepaths=filepaths[val_indices], labels=labels[val_indices], features=features[val_indices], rpeaks=None, transforms=val_transforms)

        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_set, batch_size=1, shuffle=False)

        # projector for tensorboard
        if projector:
            features = []
            label_list = []
            cnt = 0
            for (X, _), labels in train_loader:
                features.append(X.view(-1, 12*window_size))
                label_list.append(labels)
                cnt += 1
                if cnt > 20:
                    break  
            features = torch.vstack(features)
            label_list = torch.vstack(label_list)
            writer.add_embedding(features, metadata=label_list, tag = 'ecg')
            
        # writer.add_graph(model,(train_set[0][0][0].unsqueeze(dim=0).to(device), train_set[0][0][1].unsqueeze(dim=0).to(device)))
        
        # early stopping
        early_stopping_val = EarlyStopping(patience = 0, mode = 'min')
        early_stopping_score = EarlyStopping(patience = 10, mode = 'max')
    
        # training + validation loop
        print(f'Training on fold {i+1}')
        for epoch in range(1, total_epochs+1):
            # weight_histograms(writer, epoch-1, model) # adds weight histograms to tensorboard
            print(f'Epoch {epoch}/{total_epochs}')

            avg_train_loss = train(model, train_loader, loss, optimizer, use_rpeaks, device)
            writer.add_scalar('Train loss', avg_train_loss, epoch)

            avg_val_loss = val(model, val_loader, val_loss, classes, val_filenames[val_indices], window_size, window_stride, output_directory, use_rpeaks, device)
            writer.add_scalar('Val loss', avg_val_loss, epoch)

            score = run(label_filepaths[val_indices], output_filenames[val_indices], 'scores.csv')
            writer.add_scalar("Challenge Score", score, epoch)

            if(lr_scheduler is not None):
                lr_scheduler.step()

            stop = early_stopping_val(avg_val_loss) * early_stopping_score(score) # early stopping
            print(f"Train loss: {avg_train_loss}")
            print(f"Val loss: {avg_val_loss}    Early stopping counter: {early_stopping_val.counter}")
            print(f"Challenge score: {score}    Early stopping counter: {early_stopping_score.counter}")
            if(stop):
                break
            if(early_stopping_score.counter == 0): # save the best model
                torch.save(model.state_dict(), os.path.join('model_weights', f'model_weights_fold_{i+1}.pth'))

        # loading the best weights before threshold training
        model.load_state_dict(torch.load(os.path.join('model_weights', f'model_weights_fold_{i+1}.pth')))
        
        #### threshold training ####
        # get best overall threshold first
        best_score, best_threshold = 0, 0.5
        for threshold in np.linspace(0.1, 0.9, 9):
            _, _, _, _, _, _, _, _, _, _, score = evaluate_score(label_filepaths[val_indices], output_filenames[val_indices], threshold, verbose=False)
            if(score > best_score):
                best_threshold = threshold
                best_score = score
        print(f'Best overall threshold: {best_threshold}')
        # take the best overall threshold and use it to calculate individual thresholds
        model_thresholds = np.zeros(len(classes))
        for i in range(len(classes)):
            best_class_score, best_class_threshold = 0, 0.5
            for class_threshold in np.linspace(0.1, 0.9, 9):
                thresholds = np.full(len(classes), best_threshold)
                thresholds[i] = class_threshold
                _, _, _, _, _, _, _, _, _, _, class_score = evaluate_score(label_filepaths[val_indices], output_filenames[val_indices], thresholds, verbose=False)
                if(class_score > best_class_score):
                    best_class_threshold = class_threshold
                    best_class_score = class_score 
            model_thresholds[i] = best_class_threshold
        
        # saving model and threshold
        thresholds_list.append(model_thresholds)
        model_list.append(model)
        print(model_thresholds)
        run(label_filepaths[val_indices], output_filenames[val_indices], 'scores.csv', verbose=True, thresholds=model_thresholds)
        writer.close()
        break


# performs one training iteration
def train(model, train_loader, loss, optimizer, use_rpeaks, device):
    total_train_loss = 0
    with tqdm(train_loader, unit='batch') as minibatch:
        for (x, demographics), y, rpeaks in minibatch:
            x = x.to(device)
            demographics = demographics.to(device)
            y = y.float().to(device)

            optimizer.zero_grad()
            if use_rpeaks:
                y_pred = model(x, demographics, rpeaks)
            else:
                y_pred = model(x, demographics)
            train_loss = loss(y_pred, y)
            train_loss.backward()
            optimizer.step()
            total_train_loss += train_loss.item()
            minibatch.set_postfix(loss=train_loss.item())

    return total_train_loss/len(minibatch)


# performs one validation iteration and outputs the challenge score
def val(model, val_loader, loss, classes, filenames, window_size, window_stride, output_directory, use_rpeaks, device):
    total_val_loss = 0
    with tqdm(val_loader, unit='batch') as minibatch:
        for batch_num, ((x, demographics), y, rpeaks) in enumerate(minibatch):
            demographics = demographics.to(device)
            y = y.float().to(device)

            windows = []
            for i in range(0, x.shape[2]-window_size+1, window_stride):
                window = x[:, :, i: i+window_size]
                windows.append(window)
                if len(windows) > 20:
                    break
            windows = torch.vstack(windows).to(device)
            demographics = demographics.expand(windows.shape[0], demographics.shape[1])
            if use_rpeaks:
                r = []
                for i in range(windows.shape[0]):
                    r.append(rpeaks-i*window_stride)
                rpeaks = torch.vstack(r)
                y_preds = model(windows, demographics, rpeaks)
            else:
                y_preds = model(windows, demographics)
            y_preds = torch.sigmoid(y_preds)
            scores = torch.mean(y_preds, 0)
            y_label_pred = scores > 0.5 # threshold probabilities at 0.5
            val_loss = loss(scores, y.squeeze()) # calculate loss from probabilities
            total_val_loss += val_loss.item()
            scores = scores.detach().cpu().numpy()
            y_label_pred = np.array(y_label_pred.detach().cpu().numpy(), dtype=np.uint8)

            save_challenge_predictions(output_directory,filenames[batch_num],scores=scores,labels=y_label_pred,classes=classes)
            minibatch.set_postfix(loss=val_loss.item())
    return total_val_loss/len(minibatch)


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


def evaluate_score(label_filepaths, output_filenames, thresholds = None, verbose = True):
    verboseprint = print if verbose else lambda *a, **k: None

    # Define the weights, the SNOMED CT code for the normal class, and equivalent SNOMED CT codes.
    weights_file = os.path.join('utils', 'weights.csv')
    normal_class = '426783006'
    equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]

    # Load the scored classes and the weights for the Challenge metric.
    verboseprint('Loading weights...')
    classes, weights = load_weights(weights_file, equivalent_classes)

    # Load the label and output files.
    verboseprint('Loading label and output files...')
    label_files, output_files = label_filepaths, output_filenames
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

def run(label_filepaths, output_filenames, challenge_score_file, verbose = False, thresholds = None):
    classes, auroc, auprc, auroc_classes, auprc_classes, accuracy, f_measure, f_measure_classes, f_beta_measure, g_beta_measure, challenge_metric = evaluate_score(label_filepaths, output_filenames, verbose=verbose, thresholds=thresholds)
    output_string = 'AUROC,AUPRC,Accuracy,F-measure,Fbeta-measure,Gbeta-measure,Challenge metric\n{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}'.format(auroc, auprc, accuracy, f_measure, f_beta_measure, g_beta_measure, challenge_metric)
    with open(challenge_score_file, 'w') as f:
        f.write(output_string)
    return challenge_metric
