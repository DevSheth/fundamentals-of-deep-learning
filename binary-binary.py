import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from RBM import RBM
from RBM import stacked_RBM

class Dataset2(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = torch.from_numpy(self.data[idx]).float()
        label = torch.from_numpy(np.array(self.labels[idx])).type(torch.LongTensor)
        
        return image, label
    
class aux_Dataset2(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        dim = self.data.shape[1]
        image = self.data[idx].reshape(1,dim)
        
        return image

# Data Loading...
#'''
df1 = pd.read_csv("D:\\Coursework\\Sem 6\\CS6910\\Assignment 2\\Data_Set_2(Black_and_white_images)\\T_shirt.csv", header=None)
df2 = pd.read_csv("D:\\Coursework\\Sem 6\\CS6910\\Assignment 2\\Data_Set_2(Black_and_white_images)\\Trouser.csv", header=None)
df3 = pd.read_csv("D:\\Coursework\\Sem 6\\CS6910\\Assignment 2\\Data_Set_2(Black_and_white_images)\\Pullover.csv", header=None)
df4 = pd.read_csv("D:\\Coursework\\Sem 6\\CS6910\\Assignment 2\\Data_Set_2(Black_and_white_images)\\Coat.csv", header=None)
df5 = pd.read_csv("D:\\Coursework\\Sem 6\\CS6910\\Assignment 2\\Data_Set_2(Black_and_white_images)\\Sneaker.csv", header=None)

df = pd.concat([df1,df2,df3,df4,df5])

classes = ["T-shirt", "Trouser", "Pullover", "Coat", "Sneaker"]

data = np.array(df.values/255)
data_size = np.size(data, axis=0)
labels = np.zeros(data_size)
class_size = int(data_size/5)
labels[class_size:2*class_size] = np.ones(class_size)
labels[2*class_size:3*class_size] = 2*np.ones(class_size)
labels[3*class_size:4*class_size] = 3*np.ones(class_size)
labels[4*class_size:5*class_size] = 4*np.ones(class_size)

train_size = 0.8
train_indices = np.random.choice(np.arange(data_size), int(train_size*data_size), replace=False)
train_data = data[train_indices]
train_labels = labels[train_indices]
valid_indices = np.setdiff1d(np.arange(data_size), train_indices)
valid_data = data[valid_indices]
valid_labels = labels[valid_indices]
#'''

# define
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d1 = 784
l1 = 256
l2 = 64
l3 = 32

#'''
# RBM-1
BATCH_SIZE1 = 32
EPOCH1 = 20
LR1 = 1e-2
k1 = 2
rbm1 = RBM(d1, l1, DEVICE)
dataset1 = aux_Dataset2(torch.from_numpy(train_data).float())
reps1, loss4_1 = rbm1.train(dataset1, k1, EPOCH1, BATCH_SIZE1, LR=LR1)

# RBM-2
BATCH_SIZE2 = 32
EPOCH2 = 20
LR2 = 1e-2
d2 = l1
k2 = 2
rbm2 = RBM(d2, l2, DEVICE, real=False)
dataset2 = aux_Dataset2(reps1)
reps2, loss4_2 = rbm2.train(dataset2, k2, EPOCH2, BATCH_SIZE2, LR=LR2)

# RBM-3
BATCH_SIZE3 = 32
EPOCH3 = 20
LR3 = 1e-2
d3 = l2
k3 = 2
rbm3 = RBM(d3, l3, DEVICE, real=False)
dataset3 = aux_Dataset2(reps2)
reps3, loss4_3 = rbm3.train(dataset3, k3, EPOCH3, BATCH_SIZE3, LR=LR3)

# Stacked RBM
weights = []
weights.append(rbm1.w)
weights.append(rbm2.w)
weights.append(rbm3.w)

bias = []
bias.append(rbm1.c.reshape(l1))
bias.append(rbm2.c.reshape(l2))
bias.append(rbm3.c.reshape(l3))

net = stacked_RBM([d1,l1,l2,l3,len(classes)], weights, bias)
net.to(DEVICE)

PATH = "D:\\Coursework\\Sem 6\\CS6910\\Assignment 2\\RBM\\binary-binary-pretrained.pth"
torch.save(net.state_dict(), PATH)
#'''

#'''
BATCH_SIZE = 32
EPOCH = 30
LR = 1e-4

PATH = "D:\\Coursework\\Sem 6\\CS6910\\Assignment 2\\RBM\\binary-binary-pretrained.pth"
net = stacked_RBM([d1,l1,l2,l3,len(classes)])
net.to(DEVICE)
net.load_state_dict(torch.load(PATH))

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr=LR)
# optimizer1 = optim.Adam(net.fc1.parameters(), lr=1e-3)
# optimizer2 = optim.Adam(net.fc2.parameters(), lr=1e-3)
# optimizer3 = optim.Adam(net.fc3.parameters(), lr=1e-3)
# optimizer4 = optim.Adam(net.fc4.parameters(), lr=1e-3)

train_dataset = Dataset2(train_data, train_labels)
valid_dataset = Dataset2(valid_data, valid_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=1,
                        shuffle=False, num_workers=0)

loss4 = []
for epoch in range(EPOCH):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
        
        optimizer.zero_grad()
        # optimizer4.zero_grad()
        # if(epoch >= 2):
        #     optimizer1.zero_grad()
        #     optimizer2.zero_grad()
        #     optimizer3.zero_grad()
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # optimizer4.step()
        # if(epoch >= 2):
        #     optimizer1.step()
        #     optimizer2.step()
        #     optimizer3.step()
        
        running_loss += loss.item()
        
        pr = 50
        if i % pr == pr-1:
            print("[%d, %5d] loss: %.7f" % (epoch+1, i+1, running_loss/(i+1)))
            # running_loss = 0.0
    loss4.append(running_loss/(i+1))
    
print("Finished Training")
#'''

PATH = "D:\\Coursework\\Sem 6\\CS6910\\Assignment 2\\RBM\\binary-binary-pretrained-final.pth"
torch.save(net.state_dict(), PATH)

#'''
correct = 0
total = 0
confusion_matrix_train = np.zeros((5,5), dtype=np.int64)
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
confusion_matrix_valid = np.zeros((5,5), dtype=np.int64)
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
#'''