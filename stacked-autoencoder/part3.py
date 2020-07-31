"""
CS6910:  Assignment2, part 3 (dataset 2)
"""

import torch
import numpy as np
from Autoencoder import Stacked_AE
from Autoencoder import MLFFNN
import copy

train = []
validation = []
data_set = []
class_names = ["Bag","Coat", "Pullover", "Sandal", "Sneaker"]

for c in class_names:
    fname = "ds2/"+c+".csv"
    file = open(fname, 'r')
    category = []
    count = 0
    for line in file:
        example = []
        example.append([float(x)/255 for x in line.split(',')])    
        example = np.array(example).flatten()
        category.append(example)
        count+=1
        if(count<=5000):
            train.append(example)
        if(count >5000):
            validation.append(example)
    data_set.append(category)
    
    file.close()

train = np.array(train)
validation = np.array(validation)

dim = np.size(train[0],axis=0)       #28*28 images           
     
n_classes = 5
true_class = np.zeros(25000, dtype=int)
true_class[5000:9999] = 1
true_class[10000:14999] = 2
true_class[15000:19999] = 3
true_class[20000:24999] = 4
true_class = true_class.flatten()

val_class = np.zeros(5000, dtype=int)
val_class[1000:1999] = 1
val_class[2000:2999] = 2
val_class[3000:3999] = 3
val_class[4000:4999] = 4
val_class = val_class.flatten()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 50
learning_rate = 0.001
batch_size = 100
compressor = Stacked_AE(3, [dim, 256, 64, 32])
SAE_errors = compressor.stack_training(train, epochs, learning_rate, batch_size)

epochs = 50
learning_rate = 0.0005
batch_size = 100
classifier = MLFFNN([dim, 256, 64, 32, 5]) #connect output layer

# import weights from auto-encoder
for i in range(3):
    classifier.layer[i].parameters = copy.deepcopy(compressor.AANNs[i].encoder.parameters)

MLFFNN_errors = classifier.train(train, true_class, epochs, learning_rate, batch_size)
op = classifier.get_class(train, true_class)

confusion_matrix_train = np.zeros([5,5])
#row is actual, column is predicted
success = 0;
for i in range(len(op)):
    confusion_matrix_train[true_class[i]][op[i]] += 1
    if(true_class[i]==op[i]):
        success += 1
print("Train Accuracy: ", success/len(op))


op_val = classifier.get_class(validation, val_class)
confusion_matrix_val = np.zeros([5,5])
success = 0;
for i in range(len(op_val)):
    confusion_matrix_val[val_class[i]][op_val[i]] += 1
    if(val_class[i]==op_val[i]):
        success += 1
              
print("Validation Accuracy: ", success/len(op_val))

np.save('p3_SAE_losses.npy', SAE_errors)
np.save('p3_NN_losses.npy', MLFFNN_errors)