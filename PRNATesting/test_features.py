import numpy as np, os, sys
from scipy.io import loadmat
from dataset2 import ECGDataset
from utils.transform_utils import *

def load_challenge_data(filename):

    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)

    new_file = filename.replace('.mat','.hea')
    input_header_file = os.path.join(new_file)

    with open(input_header_file,'r') as f:
        header_data=f.readlines()


    return data, header_data

if __name__ == '__main__':
    input_directory = "C:\\Users\\Student\\Documents\\SmartHeartPipeline\\SmartHeartPipeline\\data"

    # input_directory = "/Volumes/macbackup/Other/physionet.org/files/challenge-2020/1.0.2/training/"

    # output_directory = sys.argv[3]
    input_files = []

    for dir, subdirs, files in os.walk(input_directory):
        for f in files:
            if os.path.isfile(os.path.join(dir, f)) and not f.lower().startswith('.') and f.lower().endswith('mat'):
                input_files.append(os.path.join(dir, f))
            
    train_transforms = Compose([])
    dataset = ECGDataset(filenames = input_files, labels = [], transforms=train_transforms)

    print(dataset.__len__())
    for i in range(dataset.__len__()):
        print(i)
        dataset.__getitem__(i)

    # add looping through, comment out