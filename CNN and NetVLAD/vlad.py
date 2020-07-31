import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

import os
import cv2
from load_dataset import dataset

class NetVLAD(nn.Module):
    def __init__(self, K, D, beta):
        super(NetVLAD, self).__init__()
        self.K = K
        self.D = D
        self.beta = beta
        self.centroids = nn.Parameter(torch.rand(K, D))
        self.conv = nn.Conv2d(D, K, kernel_size=1, bias=True)
        self.conv.weight = nn.Parameter((2.0*beta*self.centroids).unsqueeze(-1).unsqueeze(-1))
        self.conv.bias = nn.Parameter(-1*beta*self.centroids.norm(dim=1))
        
    def forward(self, x):
        N, C = x.shape[:2]  
        
        soft_assign = self.conv(x).view(N, self.K, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        
        x_flatten = x.view(N,C,-1)
        
        residual = x_flatten.expand(self.K, -1,-1,-1).permute(1,0,2,3) -\
            self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1,2,0).unsqueeze(0)
            
        residual *= soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)
        
        vlad = F.normalize(vlad, p=2, dim=2)
        vlad = vlad.view(x.size(0), -1)
        vlad = F.normalize(vlad, p=2, dim=1)
        return vlad
    
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3,4,3,padding=1)
        self.pool1 = nn.AvgPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(4,16,3,padding=1)
        self.pool2 = nn.AvgPool2d(2, stride=2)
        self.netvlad = NetVLAD(4,16,0.5)
        self.fc = nn.Linear(64, 7)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.netvlad(x)
        x = self.fc(x)
        return x

PATH = "D:\\Coursework\\Sem 6\\CS6910\\Assignment 3\\"
classes = ["030.Fish_Crow", "041.Scissor_tailed_Flycatcher", "049.Boat_tailed_Grackle", "082.Ringed_Kingfisher", 
           "103.Sayornis", "114.Black_throated_Sparrow", "168.Kentucky_Warbler"]

DEVICE = torch.device("cpu" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 10
EPOCH = 5
LR = 1e-3

full_dataset = dataset(filename=os.path.join(PATH, "train.h5"))
train_size = int(len(full_dataset)*0.8)
valid_size = len(full_dataset) - train_size
train_dataset, valid_dataset = torch.utils.data.random_split(full_dataset, [train_size, valid_size])
print("Train Data Size:", len(train_dataset))
print("Valid Data Size:", len(valid_dataset))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=0, shuffle=False)

net = CNN()

criterion = nn.CrossEntropyLoss()
mse = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=LR)

net.to(DEVICE)

for epoch in range(EPOCH):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
        
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item()
        
        pr = 10
        if i % pr == pr-1:
            print("[%d, %5d] loss: %.7f" % (epoch+1, i+1, running_loss/pr))
            running_loss = 0.0

print("Finished Training")

correct = 0
total = 0
confusion_matrix_train = np.zeros((7,7))
with torch.no_grad():
    for data in train_loader:
        inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        for i in range(predicted.shape[0]):
            confusion_matrix_train[labels[i].item()][predicted[i].item()] += 1

print('Accuracy of the network on the training data: %.2f %%' % (
    100 * correct / total))
print("Training Confusion Matrix")
print(confusion_matrix_train)

correct = 0
total = 0
confusion_matrix_valid = np.zeros((7,7))
with torch.no_grad():
    for data in valid_loader:
        inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        for i in range(predicted.shape[0]):
            confusion_matrix_valid[labels[i].item()][predicted[i].item()] += 1

print('Accuracy of the network on the validation data: %.2f %%' % (
    100 * correct / total))
print("Validation Confusion Matrix")
print(confusion_matrix_valid)