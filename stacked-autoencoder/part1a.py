"""
CS6910: Assignment2, part 1(A)
"""

import torch
import glob   
import numpy as np
from Autoencoder import MLFFNN
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

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

train = np.array(train)

true_class = []
n_classes = np.size(data_set, axis=0)
class_name = ["coast", "insidecity", "opencountry", "street", "tallbuilding"]
for i in range(n_classes):
    my_class = i
    for j in range(np.size(data_set[i], axis=0)):
        true_class.append(my_class)
    
true_class = np.array(true_class)

# use PCA for dimension reduction
pca_dim = 64
pca = PCA(n_components=pca_dim)
red_train = pca.fit_transform(train)

red_train = np.array(red_train)

train_data, val_data, train_class, val_class = train_test_split(red_train, true_class, test_size=0.20, shuffle = True)

device = torch.device("cpu")
learning_rate = 0.001
epochs = 100
batch_size = 32

layer_sizes = [pca_dim, 32, 16, 5]
model = MLFFNN(layer_sizes)
PCA_losses = model.train(train_data, train_class, epochs, learning_rate, batch_size)

op = model.get_class(train_data, train_class)
confusion_matrix_train = np.zeros([5,5])
#row is actual, column is predicted
success = 0;
for i in range(len(op)):
    confusion_matrix_train[train_class[i]][op[i]] += 1
    if(train_class[i]==op[i]):
        success += 1
        
print("Train Accuracy: ", success/len(op))


op_val = model.get_class(val_data, val_class)
confusion_matrix_val = np.zeros([5,5])    
success = 0;
for i in range(len(op_val)):
    confusion_matrix_val[val_class[i]][op_val[i]] += 1
    if(val_class[i]==op_val[i]):
        success += 1
            
print("Validation Accuracy: ", success/len(op_val))

np.save('p1a_pca_losses.npy', PCA_losses)