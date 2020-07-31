import numpy as np
import matplotlib.pyplot as plt

import os
import os.path
import h5py
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class dataset(Dataset):
    def __init__(self, filename, train=True):
        super().__init__()
        self.h5f = h5py.File(filename, "r")
        self.keys = list(self.h5f.keys())
        self.limits = np.array(self.h5f[str(0)])
        if(train):
            self.transform = transforms.Compose([
                                                    transforms.ToPILImage(),
                                                    transforms.Resize((256,256)),
                                                    #transforms.RandomHorizontalFlip(),
                                                    #transforms.RandomRotation(10),
                                                    #transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
                                                    #transforms.ColorJitter(brightness=1, contrast=1, saturation=1),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                ])
        else:
            self.transform = transforms.Compose([
                                                    transforms.ToPILImage(),
                                                    transforms.Resize((256,256)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                ])

    def __len__(self):
        return len(self.keys)-1

    def __getitem__(self, index):
        key = self.keys[index+1]
        data = np.array(self.h5f[key])
        label = 0
        # print(label, key)
        for l in self.limits:
            if(int(key) < l):
                break
            else:
                label += 1
        return self.transform(data), torch.from_numpy(np.array(label)).type(torch.LongTensor)

'''
PATH = "D:\\Coursework\\Sem 6\\CS6910\\Assignment 3\\"

all_train = True

train_dataset = dataset(filename=os.path.join(PATH, "train.h5"))
train_loader = DataLoader(train_dataset, batch_size=1, num_workers=0, shuffle=True)

if not all_train:
    valid_dataset = dataset(filename=os.path.join(PATH, "valid.h5"))
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=0, shuffle=False)

dataiter = iter(train_loader)
img, label = dataiter.next()
N, C, H, W = img.shape
plt.imshow(cv2.cvtColor(img.reshape(C, H, W).permute(1,2,0).numpy(), cv2.COLOR_BGR2RGB))
plt.show()

if not all_train:
    dataiter = iter(valid_loader)
    img, label = dataiter.next()
    N, C, H, W = img.shape
    plt.imshow(cv2.cvtColor(img.reshape(C, H, W).permute(1,2,0).numpy(), cv2.COLOR_BGR2RGB))
    plt.show()
#'''
    