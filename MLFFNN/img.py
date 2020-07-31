import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.special import softmax
import sys
#sys.path.insert(1, 'D:\\Coursework\\Sem 6\\CS6910\\Assignment 1')
import NN
from sklearn.decomposition import PCA

bird = np.load('img_feature/bird.npy')
cat = np.load('img_feature/cat.npy')
deer = np.load('img_feature/deer.npy')
ship = np.load('img_feature/ship.npy')
truck = np.load('img_feature/truck.npy')

total = np.zeros((1500, 512))

total[0:300, :] = np.copy(bird[0:300, :]);
total[300:600, :] = np.copy(cat[0:300, :]);
total[600:900, :] = np.copy(deer[0:300, :]);
total[900:1200, :] = np.copy(ship[0:300, :]);
total[1200:1500, :] = np.copy(truck[0:300, :]);

pca = PCA(n_components = 32)

pca.fit(total)
total_red = pca.transform(total)

train = np.zeros((1000,32))
train[0:200, :] = total_red[0:200, :]
train[200:400, :] = total_red[300:500, :]
train[400:600, :] = total_red[600:800, :]
train[600:800, :] = total_red[900:1100, :]
train[800:1000, :] = total_red[1200:1400, :]

truth = np.zeros((1000,5))
truth[0:200, 0] = 1
truth[200:400, 1] = 1
truth[400:600, 2] = 1
truth[600:800, 3] = 1
truth[800:1000, 4] = 1

val = np.zeros((500, 32))
val[0:100, :] = total_red[200:300, :]
val[100:200, :] = total_red[500:600, :]
val[200:300, :] = total_red[800:900, :]
val[300:400, :] = total_red[1100:1200, :]
val[400:500, :] = total_red[1400:1500, :]

val_truth = np.zeros((500, 5))
val_truth[0:100, 0] = 1
val_truth[100:200, 1] = 1
val_truth[200:300, 2] = 1
val_truth[300:400, 3] = 1
val_truth[400:500, 4] = 1

nn = NN.mlffnn(4, [32, 20, 10, 5], softmax, NN.tanh, 0, NN.der_tanh, NN.cross_entropy_grad, NN.accuracy_error)
train_err = nn.train(train, truth, 2, 0.01, 0.1, [0.9, 0.99])

#np.save('img_feature/weights_20_10' ,nn.W)
#np.save('img_feature/bias_20_10', nn.B)

train_err = np.multiply(train_err, 100)
x = np.arange(1, len(train_err)+1)
plt.plot(x, train_err)
plt.axes()
plt.xlabel('Number of Epochs')
plt.ylabel('Average Train Error (% Misclassifications)')
plt.title('Average Error after each epoch of Training')
plt.tight_layout()
plt.savefig('img_feature/train_error_2.pdf', dpi=1200)
plt.close()



val_out = np.zeros((500, 5))
confusion = np.zeros((5,5))
for i in range(500):
    val_out[i] = nn.forward(val[i]).transpose()
    x = np.argmax(val_truth[i])
    y = np.argmax(val_out[i])
    confusion[x][y] += 1
   