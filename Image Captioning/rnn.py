import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from image_caption_dataset import ImageCaption

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

class RNN(nn.Module):
    def __init__(self, d_inp, d_hid, d_out):
        super(RNN, self).__init__()
        
        self.d_inp = d_inp
        self.d_hid = d_hid
        self.d_out = d_out
        
        self.hidden = nn.Linear(d_inp+d_hid, d_hid)
        self.output = nn.Linear(d_hid, d_out)
        self.activation =  nn.Sigmoid()
        
    def forward(self, features, caption):
        features = features[0]
        caption = caption[0]
        
        h_in = features
        x_in = caption[0, :]
        #always starts with $
        
        #store vectors for each caption
        output = torch.empty(caption.size(0), self.d_out)
        output[0, :] = x_in
        
        for t in range(1, caption.size(0)):   
            combined = torch.cat((x_in, h_in), 0)
            h_out = self.activation(self.hidden(combined))
            y_out = self.activation(self.output(h_out))
            output[t,:] = y_out
            h_in = h_out
            # x_in = y_out
            x_in = caption[t,:]
        
        return output

class ImgCap(nn.Module):
    def __init__ (self, clusters, alpha, rnn_inp, rnn_hid, rnn_out, vocab, emb):
        super(ImgCap, self).__init__()
        self.conv1 = nn.Conv2d(3,4,3,padding=1)
        self.pool1 = nn.AvgPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(4,16,3,padding=1)
        self.pool2 = nn.AvgPool2d(2, stride=2)
        self.netvlad = NetVLAD(clusters, 16, alpha)
        self.fc = nn.Linear(16*clusters, rnn_hid)
        self.RNN = RNN(rnn_inp, rnn_hid, rnn_out)
        self.mse = nn.MSELoss()
        self.vocab = vocab
        self.emb = torch.from_numpy(emb).float()
        
    def forward(self, img, cap):
        x = F.relu(self.conv1(img))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        features = self.netvlad(x)
        features = self.fc(features)
        
        output = self.RNN(features, cap)
        
        loss = self.mse(output, cap[0])
        return loss
    
    def generate(self, img):
        x = F.relu(self.conv1(img))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        features = self.netvlad(x)
        features = self.fc(features)
        
        features = features[0]
        
        h_in = features
        x_in = self.emb[self.vocab["$"]]
        #always starts with $
        
        #store vectors for each caption
        output = ["$"]
        
        while True:   
            combined = torch.cat((x_in, h_in), 0)
            h_out = self.RNN.activation(self.RNN.hidden(combined))
            y_out = self.RNN.activation(self.RNN.output(h_out))
            h_in = h_out
            x_in = y_out
            word = self.similarTo(x_in)
            output.append(word)
            if word == "\n" or len(output) > 100:
                break
        
        return output
    
    def similarTo(self, vec):
        cor = np.inf
        ans = "\n"
        vec = vec.cpu().detach().numpy()
        for w in list(self.vocab.keys()):
            e = self.emb[self.vocab[w]].cpu().detach().numpy()
            val = np.linalg.norm.dot(vec - e)
            if val < cor:
                cor = val
                ans = w
        return ans
    
#'''
full_dataset = ImageCaption("./caption_glove.pth", 25)
train_size = int(len(full_dataset)*0.8)
valid_size = len(full_dataset) - train_size
train_dataset, valid_dataset = torch.utils.data.random_split(full_dataset, [train_size, valid_size])
print("Train Data Size:", len(train_dataset))
print("Valid Data Size:", len(valid_dataset))
train_loader = DataLoader(train_dataset, batch_size=1, num_workers=0, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=0, shuffle=False)
#'''

DEVICE = torch.device("cpu" if torch.cuda.is_available() else "cpu")
EPOCH = 1
LR = 1e-5

net = ImgCap(4, 0.5, 25, 32, 25, full_dataset.vocab, full_dataset.emb)

state_dict = torch.load("./vlad_2000.pth")
own_state = net.state_dict()
for name, param in state_dict.items():
    if name not in own_state:
         continue
    if isinstance(param, nn.Parameter):
        # backwards compatibility for serialized parameters
        param = param.data
    own_state[name].copy_(param)

optimizer = optim.Adam(net.parameters(), lr=LR)

net.to(DEVICE)

for epoch in range(EPOCH):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        img, cap = data[0].to(DEVICE), data[1].to(DEVICE)
        
        optimizer.zero_grad()
        
        loss = net(img, cap)
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item()
        
        pr = 100
        if i % pr == pr-1:
            print("[%d, %5d] loss: %.7f" % (epoch+1, i+1, running_loss/pr))
            running_loss = 0.0

print("Finished Training")

with torch.no_grad():
    for data in train_loader:
        img, cap = data[0].to(DEVICE), data[1].to(DEVICE)
        output = net.generate(img)
        print("Caption:", data[2])
        print("Output:", output)
        break
