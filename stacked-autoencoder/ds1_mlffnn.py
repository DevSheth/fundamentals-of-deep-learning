"""
Created on Sat May 16 12:35:19 2020

@author: hk3
CS6910:  Assignment2, part 2
"""

import glob   
import numpy as np
from Autoencoder import MLFFNN
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

train_data, val_data, train_class, val_class = train_test_split(train, true_class, test_size=0.20, shuffle = True)


epochs = 100
learning_rate = 0.0005
batch_size = 32
classifier = MLFFNN([dim, 256, 64, 32, 5])

MLFFNN_errors = classifier.train(train_data, train_class, epochs, learning_rate, batch_size)


op = classifier.get_class(train_data, train_class)
confusion_matrix_train = np.zeros([5,5])
#row is actual, column is predicted
success = 0;
for i in range(len(op)):
    confusion_matrix_train[train_class[i]][op[i]] += 1
    if(train_class[i]==op[i]):
        success += 1

print("Train Accuracy: ", success/len(op))
    

op_val = classifier.get_class(val_data, val_class)
confusion_matrix_val = np.zeros([5,5])    

success = 0;
for i in range(len(op_val)):
    confusion_matrix_val[val_class[i]][op_val[i]] += 1
    if(val_class[i]==op_val[i]):
        success += 1
            
print("Validation Accuracy: ", success/len(op_val))

np.save('ds1_NN_losses.npy', MLFFNN_errors)