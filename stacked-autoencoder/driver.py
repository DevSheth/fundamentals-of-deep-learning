#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 11:15:44 2020

@author: hk3
testing ground of different functions and designs
"""
import torch
import glob   
import numpy as np
from Autoencoder import AE
from Autoencoder import Stacked_AE
from torch import nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset

train = []
data_set = []
path = 'ds1/*'   
folders = glob.glob(path)   

for folder in folders:   
    category = []
    class_path = folder + "/*"
    files = glob.glob(class_path)
    for file in files:
        f = open(file, 'r')
        example = []
        for line in f:
            example.append([float(x) for x in line.split(' ')])
            
        example = np.array(example).flatten()
        train.append(example)
        category.append(example)
        f.close()
    data_set.append(np.array(category))

dim = len(train[0])

true_class = []
n_classes = np.size(data_set, axis=0)
for i in range(n_classes):
    my_class = np.zeros(n_classes)
    my_class[i] = 1
    for j in range(np.size(data_set[i], axis=0)):
        true_class.append(my_class)
    
true_class = np.array(true_class)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_data = np.array(train)

#x_train = Variable(torch.from_numpy(train)).type(torch.FloatTensor)
#y_train = Variable(torch.from_numpy(train)).type(torch.FloatTensor)


learning_rate = 0.001
epochs = 100
batch_size = 50

model = Stacked_AE(1, [dim, 400])
model.stack_training(train_data, epochs, learning_rate, 50)

encoded_data = model.encoding(1, train_data)
#model = AE(dim, 32)
#model.train(train_data, epochs, learning_rate, batch_size)
#encoded_data = model.encoding(train_data)



