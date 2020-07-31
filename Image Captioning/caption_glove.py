import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from glove import GloVe

#'''
vocab = {}
N = 0
file = open("./Task 1/captions.txt", "r")
captions = []
images = {}
for i, line in enumerate(file):
    image, caption = line.split("\t")
    [caption, _] = caption.split("\n")
    [image, _] = image.split("#")
    words = caption.split(" ")
    words.append("\n")
    words.insert(0, "$")
    captions.append(words)
    images.append(image)
    for word in words:
        word = word.lower()
        if word not in vocab:
            vocab[word] = N
            N += 1        
file.close()

X = np.zeros((N, N))
k = 10

for words in captions:
    n = len(words)
    for i in range(n):
        for j in range(max(0,i-k), min(n, i+k+1)):
            if not vocab[words[i].lower()] == vocab[words[j].lower()]:
                X[vocab[words[i].lower()]][vocab[words[j].lower()]] += 1/(abs(i-j))
#'''

'''
K = 25
G = GloVe(X, K)

EPOCH = 100
BS = 500
LR = 1e-2

optimizer = optim.Adam(G.parameters(), lr=LR)
mse = nn.MSELoss()

(x_orig, y_orig) = np.where(X > 0)
count = x_orig.size

losses = []
for epoch in range(EPOCH):
    running_loss = 0
    idx = np.random.permutation(count)
    x = x_orig[idx]
    y = y_orig[idx]
    for i in range(int(count/BS)):
        optimizer.zero_grad()
        loss = G(x[BS*i:BS*(i+1)], y[BS*i:BS*(i+1)])/BS
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pr = 100
        if i % pr == pr-1:
            print("Epoch: %d, Batch: %d, Loss: %.10f" % ((epoch+1), (i+1), running_loss/(i+1)))
    losses.append(running_loss/(i+1))
    print("Epoch: %d, Loss: %.10f" % ((epoch+1), running_loss/(i+1)))
    
x = np.arange(1,EPOCH+1)
plt.plot(x, losses)
plt.axes()
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.title('Loss v Epochs plot for the GloVe model training on captions')
plt.tight_layout()
plt.savefig('./loss_plot.png')
plt.close()
#'''
