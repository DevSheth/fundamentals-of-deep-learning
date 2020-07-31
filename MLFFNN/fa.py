import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.special import softmax
import sys
sys.path.insert(1, 'D:\\Coursework\\Sem 6\\CS6910\\Assignment 1')
import NN

train = []
file = open('D:\\Coursework\\Sem 6\\CS6910\\Assignment 1\\Function Approximation\\Team37\\train100.txt', 'r')
for line in file:
    train.append(np.array([float(x) for x in line.split(' ')]))
train = np.array(train)
data = train[:, 0:2]
truth = train[:, 2:3]

nn = NN.mlffnn(4, [2,50,50,1], NN.linear, NN.tanh, NN.der_linear, 
               NN.der_tanh, NN.squared_error_grad, NN.avg_error)
train_err = nn.train(data, truth, 1, 0.0002, 0.9, (0.9, 0.99))

x = np.arange(1, len(train_err)+1)
plt.plot(x, train_err)
plt.axes()
plt.xlabel('Number of Epochs')
plt.ylabel('Average Train Error')
plt.title('Average Error after each epoch of Training')
plt.tight_layout()
plt.savefig('D:\\Coursework\\Sem 6\\CS6910\\Assignment 1\\Function Approximation\\train_error.pdf', dpi=1200)
plt.close()

out = np.zeros(100)
for i in range(100):
    out[i] = nn.forward(data[i])

plt.scatter(out, truth, s=2)
r1 = int(np.min(truth))
r2 = int(np.max(truth))
plt.plot(range(r1, r2), range(r1, r2), color = 'red')
plt.xlabel('Model output')
plt.ylabel('Ground truth')
plt.title('Scatter plot for training data')
plt.tight_layout()
plt.savefig('D:\\Coursework\\Sem 6\\CS6910\\Assignment 1\\Function Approximation\\scatter_train.pdf', dpi=1200)
plt.close()

val = []
file = open('D:\\Coursework\\Sem 6\\CS6910\\Assignment 1\\Function Approximation\\Team37\\val.txt', 'r')
for line in file:
    val.append(np.array([float(x) for x in line.split(' ')]))
val = np.array(val)
data_v = val[:, 0:2]
truth_v = val[:, 2:3]

out_v = np.zeros(300)
err = 0
for i in range(300):
    out_v[i] = nn.forward(data_v[i])
    err += NN.avg_error(truth_v[i], out_v[i])
print(err/300)
'''
plt.scatter(out_v, truth_v, s=2)
r1 = int(np.min(truth_v))
r2 = int(np.max(truth_v))
plt.plot(range(r1, r2), range(r1, r2), color = 'red')
plt.xlabel('Model output')
plt.ylabel('Ground truth')
plt.title('Scatter plot for validation data')
plt.tight_layout()
plt.savefig('D:\\Coursework\\Sem 6\\CS6910\\Assignment 1\\Function Approximation\\scatter_val.pdf', dpi=1200)
plt.close()

x1 = np.min(data[:,0])
x2 = np.max(data[:,0])
y1 = np.min(data[:,1])
y2 = np.max(data[:,1])
X, Y = np.meshgrid(np.linspace(x1,x2,100), np.linspace(y1,y2,100))
Z = np.copy(X)
for i in range(100):
    for j in range(100):
        Z[i][j] = nn.forward([X[i][j], Y[i][j]])

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(data[:,0], data[:,1], truth[:,0], s=2, color='red')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')
ax.set_title('Desired points and approximated function surface')
ax.view_init(15, -210)
plt.savefig('D:\\Coursework\\Sem 6\\CS6910\\Assignment 1\\Function Approximation\\3D_contour.pdf', dpi=1200)
plt.close()
#'''