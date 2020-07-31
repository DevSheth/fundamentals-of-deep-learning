"""
Multi Layer Feed Forward Neural Network Implementation 
"""
import numpy as np
import random

class mlffnn:
    def __init__(self, layers, layer_nodes, act_out, act_layer, der_out, der_layer, error_grad, error):
        '''
        Initilaizing all the variables associated with the network.

        Parameters
        ----------
        layers : Total number of layers
        layer_nodes : List containing number of nodes at each layer.
        act_out : Activation function used at output node.
        act_layer : Activation function used at intermediate layers.
        der_out : Derivative of activation functions at the output layer.
        der_layer : Derivative of activation functions at intermediate layers.
        error_grad : Gradient of Error function defined as per task.
        error : Error function defined as per task.

        Returns
        -------
        None.

        '''
        self.L = layers-1
        self.L_dims = layer_nodes
        self.W = []
        self.delta_W = []
        self.H = []
        self.delta_H = []
        self.A = []
        self.delta_A = []
        self.B = []
        self.delta_B = []
        self.act_out = act_out
        self.act_layer = act_layer
        self.der_out = der_out
        self.der_layer = der_layer
        self.error_grad = error_grad
        self.error = error
        for i in range(self.L):
            self.W.append((np.random.rand(layer_nodes[i+1], layer_nodes[i])-0.5)*2)
            self.delta_W.append(np.zeros((layer_nodes[i+1], layer_nodes[i])))
            self.H.append(np.array(np.zeros(layer_nodes[i+1])).transpose())
            self.delta_H.append(np.array(np.zeros(layer_nodes[i+1])).transpose())
            self.A.append(np.array(np.zeros(layer_nodes[i+1])).transpose())
            self.delta_A.append(np.array(np.zeros(layer_nodes[i+1])).transpose())
            self.B.append(np.array([(np.random.rand(layer_nodes[i+1])-0.5)*2]).transpose())
            self.delta_B.append(np.array(np.zeros(layer_nodes[i+1])).transpose())
        self.W = np.array(self.W)
        self.delta_W = np.array(self.delta_W)
        self.H = np.array(self.H)
        self.delta_H = np.array(self.delta_H)
        self.A = np.array(self.A)
        self.delta_A = np.array(self.delta_A)
        self.B = np.array(self.B)
        self.delta_B = np.array(self.delta_B)
        # only for image data to get same initialized weights
        #self.W = np.load('img_feature/weights_20_10.npy', allow_pickle = True)
        #self.B = np.load('img_feature/bias_20_10.npy', allow_pickle = True)
        
    def forward(self, x):
        '''
        Calculates the forward pass of the MLFFNN

        Parameters
        ----------
        x : The input vector to the Neural Network.

        Returns
        -------
        y : The output vector of the Neural Network.

        '''
        x = np.array([x]).transpose()
        self.A[0] = np.dot(self.W[0], x)
        self.H[0] = self.act_layer(self.A[0])
        for k in range(1, self.L-1):
            self.A[k] = np.dot(self.W[k], self.H[k-1]) + self.B[k]
            self.H[k] = self.act_layer(self.A[k])   
        self.A[self.L-1] = np.dot(self.W[self.L-1], self.H[self.L-2]) + self.B[self.L-1]
        self.H[self.L-1] = self.act_out(self.A[self.L-1])
        return self.H[self.L-1]
    
    def backprog(self, x, y):
        '''
        Calculates the back propogation errors for each sample.

        Parameters
        ----------
        x : Input to the Neural Network.
        y : Ground truth of the data.

        Returns
        -------
        None.

        '''
        y = np.array([y]).transpose()
        self.delta_A[self.L-1] = self.error_grad(y, self.forward(x), self.der_out)
        for i in range(self.L-1):
            k = self.L-i-1
            self.delta_W[k] = np.dot(self.delta_A[k], self.H[k-1].transpose())
            self.delta_B[k] = np.copy(self.delta_A[k])
            self.delta_H[k-1] = np.dot(self.W[k].transpose(), self.delta_A[k])
            self.delta_A[k-1] = self.delta_H[k-1] * self.der_layer(self.A[k-1])
        self.delta_W[0] = np.dot(self.delta_A[0], np.array([x]))
        self.delta_B[0] = np.copy(self.delta_A[0])
            
    def train(self, data, truth, opt_scheme, eta, alpha, rho):
        '''
        Trains the Neural Network on the given data-set.

        Parameters
        ----------
        data : Input data as a numpy array.
        truth : Ground truth corresponding to the input data.
        opt_scheme : Pass 0,1 or 2 as per the below scheme:-
                     0 -> Delta Rule
                     1 -> Generalized Delta Rule
                     2 -> Adam
        eta : Learning parameter.
        alpha : Used in gen delta rule, -1 o.w.
        rho : Pair used in Adam, -1 o.w.

        Returns
        -------
        The Average Error at each epoch in a numpy array.

        '''
        size = np.size(data, axis=0)
        Del_W = np.multiply(self.W, 0)
        Del_B = np.multiply(self.B, 0)
        Q_W = np.multiply(self.W, 0)
        Q_B = np.multiply(self.B, 0)
        R_W = np.multiply(self.W, 0)
        R_B = np.multiply(self.B, 0)
        avg_error = []
        ep = -1
        run = True
        
        for ep in range(1500):
        #while(run):
            #ep += 1
            err = 0
            idx = np.arange(size)
            random.shuffle(idx)
            for i in idx:
                x = np.copy(data[i])
                y = np.copy(truth[i])
                self.backprog(x, y)
                err += self.error(y, self.H[self.L-1])
                # Delta Rule
                if(opt_scheme == 0):
                    self.W = self.W - np.multiply(self.delta_W, eta)
                    self.B = self.B - np.multiply(self.delta_B, eta)
                # Generalized Delta Rule
                elif(opt_scheme == 1):
                    Del_W = - np.multiply(self.delta_W, eta) + np.multiply(Del_W, alpha)
                    Del_B = - np.multiply(self.delta_B, eta) + np.multiply(Del_B, alpha)
                    self.W = self.W + Del_W
                    self.B = self.B + Del_B
                # Adam
                elif(opt_scheme == 2):
                    Q_W = np.multiply(Q_W,rho[0]) + np.multiply(self.delta_W,(1-rho[0]))
                    Q_hat_W = np.divide(Q_W,(1 - rho[0]**(i+1)))
                    
                    Q_B = np.multiply(Q_B,rho[0]) + np.multiply(self.delta_B,(1-rho[0]))
                    Q_hat_B = np.divide(Q_B,(1 - rho[0]**(i+1)))
                    
                    R_W = np.multiply(R_W,rho[1]) + np.multiply((self.delta_W*self.delta_W), (1-rho[1]))
                    R_hat_W = np.divide(R_W,(1 - rho[1]**(i+1)))
                    
                    R_B = np.multiply(R_B,rho[1]) + np.multiply((self.delta_B*self.delta_B), (1-rho[1]))
                    R_hat_B = np.divide(R_B,(1 - rho[1]**(i+1)))
                    
                    for j in range(self.L):
                        R_hat_W[j] = np.sqrt(R_hat_W[j])
                        R_hat_B[j] = np.sqrt(R_hat_B[j])
                    
                    self.W = self.W - eta*(Q_hat_W/(0.00001 + R_hat_W))
                    self.B = self.B - eta*(Q_hat_B/(0.00001 + R_hat_B))
            err /= size
            avg_error.append(err)
            print("epoch", (ep+1), ":", err)
            l = np.min([ep+1, 100])
            stop = 0
            if(l == 100):
                #for i in range(l-1):
                #    stop += abs(avg_error[ep-i] - avg_error[ep-1-i])
                #stop /= (l-1)
                stop = abs(avg_error[ep] - avg_error[ep+1-l])
                if(stop/avg_error[ep] < 0.01):
                    run = False
        return avg_error

def linear(x):
    return x

def der_linear(x):
    return 1 

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def tanh(x):
    a = np.exp(0.1*x)
    b = np.exp(-0.1*x)
    return (a-b)/(a+b)

def der_tanh(x):
    c = tanh(x)
    return 0.1*(1 - c*c)

def der_sigmoid(x):
    c = sigmoid(x)
    return c*(1-c)

def softmax(x):
    y = np.exp(x/0.1)
    return y/np.sum(y)

def cross_entropy(y, y_hat):
    return -np.sum(y*np.log(y_hat))

def accuracy_error(y, y_hat):
    if(np.argmax(y) == np.argmax(y_hat)):
        return 0
    else:
        return 1

def cross_entropy_grad(y, y_hat, der_out):
    # Use only with softmax activation at the output layer
    return -(y - y_hat)

def squared_error(y, y_hat):
    return 0.5*(np.linalg.norm(y - y_hat)**2)

def avg_error(y, y_hat):
    return np.linalg.norm(y-y_hat, 1)

def squared_error_grad(y, y_hat, der_out):
    return -(y-y_hat)*der_out(y_hat)