import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torch import Tensor
from torch.jit.annotations import Optional, Tuple

import os
import cv2
from load_dataset import dataset

class MLFFNN(nn.Module):
    def __init__(self, layers):
        super(MLFFNN, self).__init__()
        self.fc1 = nn.Linear(layers[0], layers[1])
        self.fc2 = nn.Linear(layers[1], layers[2])
        self.fc3 = nn.Linear(layers[2], 7)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def vgg_forward(vgg, x):
    x = vgg.features(x)
    x = vgg.avgpool(x)
    x = torch.flatten(x, 1)
    return x

def goog_forward(goog, x):
    # type: (Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]
    # N x 3 x 224 x 224
    x = goog.conv1(x)
    # N x 64 x 112 x 112
    x = goog.maxpool1(x)
    # N x 64 x 56 x 56
    x = goog.conv2(x)
    # N x 64 x 56 x 56
    x = goog.conv3(x)
    # N x 192 x 56 x 56
    x = goog.maxpool2(x)

    # N x 192 x 28 x 28
    x = goog.inception3a(x)
    # N x 256 x 28 x 28
    x = goog.inception3b(x)
    # N x 480 x 28 x 28
    x = goog.maxpool3(x)
    # N x 480 x 14 x 14
    x = goog.inception4a(x)
    # N x 512 x 14 x 14

    x = goog.inception4b(x)
    # N x 512 x 14 x 14
    x = goog.inception4c(x)
    # N x 512 x 14 x 14
    x = goog.inception4d(x)
    # N x 528 x 14 x 14

    x = goog.inception4e(x)
    # N x 832 x 14 x 14
    x = goog.maxpool4(x)
    # N x 832 x 7 x 7
    x = goog.inception5a(x)
    # N x 832 x 7 x 7
    x = goog.inception5b(x)
    # N x 1024 x 7 x 7

    x = goog.avgpool(x)
    # N x 1024 x 1 x 1
    x = torch.flatten(x, 1)
    # N x 1024
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

vgg16 = models.vgg16(pretrained=True)
vgg16.eval()
googlenet = models.googlenet(pretrained=True)
googlenet.eval()

model = "vgg"

if model == "vgg":
    net = MLFFNN([25088, 4096, 1024])
elif model == "googlenet":
    net = MLFFNN([1024, 512, 128])

criterion = nn.CrossEntropyLoss()
mse = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=LR)

net.to(DEVICE)
googlenet.to(DEVICE)
vgg16.to(DEVICE)

for epoch in range(EPOCH):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
        
        optimizer.zero_grad()
        
        with torch.no_grad():
            if model == "vgg":
                inputs = vgg_forward(vgg16, inputs)
            elif model == "googlenet":
                inputs = goog_forward(googlenet, inputs)
        
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

        if model == "vgg":
            inputs = vgg_forward(vgg16, inputs)
        elif model == "googlenet":
            inputs = goog_forward(googlenet, inputs)
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

        if model == "vgg":
            inputs = vgg_forward(vgg16, inputs)
        elif model == "googlenet":
            inputs = goog_forward(googlenet, inputs)
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
