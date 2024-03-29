#!/usr/bin/env python

import os, sys
from train_12ECG_classifier import train_12ECG_classifier

if __name__ == '__main__':
    # Parse arguments.
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    config_file = sys.argv[3]

    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)
    if not os.path.isdir('model_weights'):
        os.mkdir('model_weights')

    print('Running training code...')

    train_12ECG_classifier(input_directory, output_directory, config_file)

    print('Done.')
