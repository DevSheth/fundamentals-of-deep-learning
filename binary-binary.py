import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from RBM import RBM

class Dataset2(Dataset):
    def __init__(self, pandas_frame, labels):
        self.data = pandas_frame.values/255
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = torch.from_numpy(self.data[idx]).reshape(1,784).float()
        label = torch.from_numpy(np.array(int(idx/6000))).float()
        #label = self.labels[lb]
        
        return image, label

# define
BATCH_SIZE = 50
EPOCH = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df1 = pd.read_csv("D:\\Coursework\\Sem 6\\CS6910\\Assignment 2\\Data_Set_2(Black_and_white_images)\\T_shirt.csv", header=None)
df2 = pd.read_csv("D:\\Coursework\\Sem 6\\CS6910\\Assignment 2\\Data_Set_2(Black_and_white_images)\\Trouser.csv", header=None)
df3 = pd.read_csv("D:\\Coursework\\Sem 6\\CS6910\\Assignment 2\\Data_Set_2(Black_and_white_images)\\Pullover.csv", header=None)
df4 = pd.read_csv("D:\\Coursework\\Sem 6\\CS6910\\Assignment 2\\Data_Set_2(Black_and_white_images)\\Coat.csv", header=None)
df5 = pd.read_csv("D:\\Coursework\\Sem 6\\CS6910\\Assignment 2\\Data_Set_2(Black_and_white_images)\\Sneaker.csv", header=None)

df = pd.concat([df1,df2,df3,df4,df5])

labels = ["T-shirt", "Trouser", "Pullover", "Coat", "Sneaker"]

var = torch.from_numpy(np.var(df.values/255, axis=0)).float()

dataset_2 = Dataset2(pandas_frame=df, labels=labels)

dataloader = DataLoader(dataset_2, batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=0)

mse = nn.MSELoss()

#'''
d = 784
l = 256
k = 2
rbm = RBM(d, l, DEVICE)

for epoch in range(EPOCH):
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        image, label = data[0].to(DEVICE), data[1].to(DEVICE)
        
        rbm.contrastive_divergence(k, image, optim="Adam")
        
        output = rbm.get_output(image)
        loss = mse(image, output)*255
        running_loss += loss.item()
        pr = 20
        if i % pr == pr-1:
            print("[%d, %5d] loss: %.4f" % (epoch+1, i+1, running_loss/pr))
            running_loss = 0.0

print("Finished Training")
#'''