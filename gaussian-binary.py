import numpy as np
import pandas as pd
import glob
import copy

import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.utils.data import TensorDataset
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from RBM import RBM

BATCH_SIZE = 100
EPOCH = 10
DEVICE = torch.device("cpu" if torch.cuda.is_available() else "cpu")

train = []
data_set = []
path = "D:\\Coursework\\Sem 6\\CS6910\\Assignment 2\\Data_Set_1(Colored_Images)\\data\\*"
folders = glob.glob(path)   

for folder in folders:   
    category = []
    class_path = folder + "\\*"
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
train = np.array(train).reshape(1726, 1, 828)

true_class = []
n_classes = np.size(data_set, axis=0)
class_name = ["coast", "insidecity", "opencountry", "street", "tallbuilding"]
for i in range(n_classes):
    my_class = i
    for j in range(np.size(data_set[i], axis=0)):
        true_class.append(my_class)
    
true_class = np.array(true_class)

var = torch.from_numpy(np.var(train, axis=0)).float()

x_train = Variable(torch.from_numpy(train)).type(torch.FloatTensor)
y_train = Variable(torch.from_numpy(true_class)).type(torch.FloatTensor)

dataset_1 = TensorDataset(x_train, y_train)

dataloader = DataLoader(dataset_1, batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=0)

mse = nn.MSELoss()

classes = ["coast", "insidecity", "opencountry", "street", "tallbuilding"]

#'''
d = 828
l = 256
k = 2
rbm = RBM(d, l, DEVICE, gaussian=True, var=var)

for epoch in range(EPOCH):
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        image, label = data[0].to(DEVICE), data[1].to(DEVICE)
        
        rbm.contrastive_divergence(k, image, optim="Delta", lr=1e-10)
        
        output = rbm.get_output(image)
        loss = mse(image, output)
        running_loss += loss.item()
        pr = 1
        if i % pr == pr-1:
            print("[%d, %5d] loss: %.4f" % (epoch+1, i+1, running_loss/pr))
            running_loss = 0.0

print("Finished Training")
#'''