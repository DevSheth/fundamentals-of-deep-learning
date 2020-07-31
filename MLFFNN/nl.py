import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
#from scipy.special import softmax
import sys
sys.path.insert(1, 'D:\\Coursework\\Sem 6\\CS6910\\Assignment 1')
import NN

Train = []
file = open('D:\\Coursework\\Sem 6\\CS6910\\Assignment 1\\2D Non-Linear\\traingroup37.csv', 'r')
for line in file:
    Train.append(np.array([float(x) for x in line.split(',')]))
Train = np.array(Train)
temp = Train[Train[:,2].argsort()]
Train = np.copy(temp)

train = np.copy(Train[0:315,:])
train[105:210, :] = np.copy(Train[151:256, :])
train[210:315, :] = np.copy(Train[301:406, :])
val = np.zeros((136,3))
val[0:46, :] = np.copy(Train[105:151, :])
val[46:91, :] = np.copy(Train[256:301, :])
val[91:136, :] = np.copy(Train[406:451, :])

data = np.zeros((315, 2))
data[:, 0:2] = np.copy(train[:,0:2])
#data[:, 2] = data[:, 0]**2
#data[:, 3] = data[:, 1]**2
#data[:, 4] = data[:, 0]*data[:, 1]
truth = np.zeros((len(data), 3))
for i in range(len(data)):
    truth[i][int(train[:,2][i])] = 1
    
nn = NN.mlffnn(4, [2,10,10,3], NN.softmax, NN.tanh, NN.der_linear, 
               NN.der_tanh, NN.cross_entropy_grad, NN.accuracy_error)
train_err = nn.train(data, truth, 1, 0.002, 0.1, (0.9, 0.99))

data_v = val[:,0:2]
truth_v = np.zeros((len(data_v), 3))
for i in range(len(data_v)):
    truth_v[i][int(val[:,2][i])] = 1
err = 0
for i in range(len(data_v)):
    err += NN.accuracy_error(truth_v[i], nn.forward(data_v[i]))
print(err/len(data_v))

'''
x1 = int(np.min(data[:,0])-5)
x2 = int(np.max(data[:,0])+5)
y1 = int(np.min(data[:,1])-5)
y2 = int(np.max(data[:,1])+5)
X, Y = np.meshgrid(np.linspace(x1,x2,1000), np.linspace(y1,y2,1000))
Z1 = np.copy(X)
Z2 = np.copy(X)
Z = np.copy(X)

for i in range(1000):
    for j in range(1000):
        nn.forward([X[i][j], Y[i][j]])
        Z[i][j] = np.argmax(nn.H[2])


for k in range(10):
    for i in range(100):
        for j in range(100):
            nn.forward([X[i][j], Y[i][j]])
            Z1[i][j] = nn.H[0][k]
            Z2[i][j] = nn.H[1][k]
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z1, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Output at a node in Hidden Layer 1')
    ax.set_title('Desired points and approximated function surface')
    ax.view_init(15, -210)
    plt.savefig('D:\\Coursework\\Sem 6\\CS6910\\Assignment 1\\2D Non-Linear\\3D_contour_ep1_HL1_{}.pdf'.format(k+1), dpi=1200)
    plt.close()
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z2, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Output at a node in Hidden Layer 1')
    ax.set_title('Desired points and approximated function surface')
    ax.view_init(15, -210)
    plt.savefig('D:\\Coursework\\Sem 6\\CS6910\\Assignment 1\\2D Non-Linear\\3D_contour_ep1_HL2_{}.pdf'.format(k+1), dpi=1200)
    plt.close()
    
for k in range(3):
    for i in range(100):
        for j in range(100):
            nn.forward([X[i][j], Y[i][j]])
            Z[i][j] = nn.H[2][k]
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Output at a node in Hidden Layer 1')
    ax.set_title('Desired points and approximated function surface')
    ax.view_init(15, -210)
    plt.savefig('D:\\Coursework\\Sem 6\\CS6910\\Assignment 1\\2D Non-Linear\\3D_contour_ep1_OP_{}.pdf'.format(k+1), dpi=1200)
    plt.close()
#'''