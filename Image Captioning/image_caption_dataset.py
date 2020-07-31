import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import os
import os.path
import glob
import cv2
from glove import GloVe

class ImageCaption(Dataset):
    def __init__(self, path, K):
        vocab = {}
        N = 0
        file = open("./Task 1/captions.txt", "r")
        file2 = open("./Task 1/image_names.txt", "r")
        lines2 = file2.readlines()
        for i,l in enumerate(lines2):
            [l,_] = l.split('\n')
            lines2[i] = l
        captions = []
        images = {}
        for i, line in enumerate(file):
            image, caption = line.split("\t")
            [caption, _] = caption.split("\n")
            [image, _] = image.split("#")
            if image in lines2:
                if image not in images:
                    images[image] = []
                images[image].append(caption)
            words = caption.split(" ")
            words.append("\n")
            words.insert(0, "$")
            captions.append(words)
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
        
        G = GloVe(X, K)
        G.load_state_dict(torch.load(path))
        
        self.emb = G.embeddings()
        self.K = K
        self.vocab = vocab
        self.images = images
        self.keys = list(self.images.keys())
        self.transform = transforms.Compose([
                                                transforms.ToPILImage(),
                                                transforms.Resize((256,256)),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                            ])
        self.to_torch = transforms.Compose([transforms.ToTensor()])
        
    def __len__(self):
        return len(self.images)*5
    
    def __getitem__(self, idx):
        i = idx // 5
        j = idx % 5
        path = os.path.join("./Task 1/Images", self.keys[i])
        img = cv2.imread(path)
        img = np.array(img)
        caption = self.images[self.keys[i]][j]
        words = caption.split()
        words.insert(0, "$")
        words.append("\n")
        cap = np.zeros((len(words), self.K))
        for w, word in enumerate(words):
            cap[w] = self.emb[self.vocab[word.lower()]]
        return self.transform(img), self.to_torch(cap).reshape(len(words), self.K).float(), caption
