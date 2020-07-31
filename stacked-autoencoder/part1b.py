"""
CS6910: Assignment2, part 1(B)
"""

import torch
import glob   
import numpy as np
from Autoencoder import AE
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# use AutoEncoder for dimension reduction
print("training auto-encoder")
red_dim = 64
model = AE(dim, red_dim)
learning_rate = 0.004
epochs = 100
batch_size = 32
AE_training_errors = model.train(train, epochs, learning_rate, batch_size)
red_train = model.get_encoding(train) 

train_data, val_data, train_class, val_class = train_test_split(red_train, true_class, test_size=0.20, shuffle = True)


print("training classifier")
learning_rate = 0.003
batch_size =  32
epochs = 200
layer_sizes = [red_dim, 32, 16, 5]
classifier = MLFFNN(layer_sizes)
MLFFNN_training_errors = classifier.train(train_data, train_class, epochs, learning_rate, batch_size)
op = classifier.get_class(train_data, train_class)

confusion_matrix_train = np.zeros([5,5])
#row is actual, column is predicted

success = 0;
for i in range(len(op)):
    confusion_matrix_train[train_class[i]][op[i]] += 1
    if(train_class[i]==op[i]):
        success += 1
        
print("Training Accuracy: ", success/len(op))


op_val = classifier.get_class(val_data, val_class)
confusion_matrix_val = np.zeros([5,5])    
success = 0;
for i in range(len(op_val)):
    confusion_matrix_val[val_class[i]][op_val[i]] += 1
    if(val_class[i]==op_val[i]):
        success += 1
            
print("Validation Accuracy: ", success/len(op_val))


##observation: higher LR for AE and lower LR for attcached MLFFNN for better results
## increasing MLFFNN epochs reduces train error, but validation accuracy decreases 

np.save('p1b_AE_losses.npy', AE_training_errors)
np.save('p1b_NN_losses.npy', MLFFNN_training_errors)