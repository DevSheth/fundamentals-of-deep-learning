import numpy as np
import pandas as pd

import os
import os.path
import h5py
import cv2
import glob
import tqdm

PATH = "D:\\Coursework\\Sem 6\\CS6910\\Assignment 3\\"

classes = ["030.Fish_Crow", "103.Sayornis", "082.Ringed_Kingfisher", "049.Boat_tailed_Grackle", 
           "168.Kentucky_Warbler", "041.Scissor_tailed_Flycatcher", "114.Black_throated_Sparrow"]

image_ids = pd.read_csv("D:\\Coursework\\Sem 6\\CS6910\\Assignment 3\\CUB_200_2011\\images.txt", sep=' ', header=None)
train_test_split = pd.read_csv("D:\\Coursework\\Sem 6\\CS6910\\Assignment 3\\CUB_200_2011\\train_test_split.txt", sep=' ', header=None)

get_id = {image_ids.values[i, 1]: image_ids.values[i, 0] for i in range(len(image_ids.values))}
is_train = {train_test_split.values[i, 0]: train_test_split.values[i, 1] for i in range(len(train_test_split.values))}

folders = glob.glob(os.path.join(PATH, "CUB_200_2011\\images\\*"))

train_limits = []
valid_limits = []

all_train = True

with h5py.File(os.path.join(PATH, "train.h5"), "w") as h5f_train:
    with h5py.File(os.path.join(PATH, "valid.h5"), "w") as h5f_valid:
        train_size = 1
        valid_size = 1
        for folder in folders:
            FOLDER = os.path.basename(folder)
            files = sorted(glob.glob(os.path.join(folder, "*.jpg")))
            for file in tqdm.tqdm(files):
                FILE = os.path.basename(file)
                KEY = FOLDER + "/" + FILE
                img = cv2.imread(file)
                h, w, c = img.shape
                img = np.array(img)
                ID = get_id[KEY]
                if(is_train[ID] == 1 or all_train):
                    h5f_train.create_dataset(str(train_size), data=img)
                    train_size += 1
                else:
                    h5f_valid.create_dataset(str(valid_size), data=img)
                    valid_size += 1
            train_limits.append(train_size)
            valid_limits.append(valid_size)
        h5f_train.create_dataset(str(0), data=np.array(train_limits))
        h5f_valid.create_dataset(str(0), data=np.array(valid_limits))
