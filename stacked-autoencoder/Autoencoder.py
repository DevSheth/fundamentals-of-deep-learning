import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset
import numpy as np
           

class AE(nn.Module):
    def __init__(self, d_inp, d_hid):
        """
        Parameters
        ----------
        d_inp : dimension of input layer
        d_hid : dimension of hidden layer

        Returns
        -------
        None.
        """
        super(AE, self).__init__()
        self.d_inp = d_inp
        self.d_hid = d_hid
        self.encoder = nn.Linear(d_inp, d_hid)
        self.decoder = nn.Linear(d_hid, d_inp)
        
    def forward(self, x):
        x = F.relu(self.encoder(x))
        x = F.relu(self.decoder(x))
        return x
    def encode(self, x):
        x = F.relu(self.encoder(x))
        return x
    
    def train(self, train_data, epochs, learning_rate, batch_size):   
        x_train = Variable(torch.from_numpy(train_data)).type(torch.FloatTensor)
        y_train = Variable(torch.from_numpy(train_data)).type(torch.FloatTensor)
        
        trn = TensorDataset(x_train, y_train)
        trn_dataloader = torch.utils.data.DataLoader(trn, batch_size=batch_size, shuffle=True, num_workers=2)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss_func = nn.MSELoss()
        print(self)
                
        losses = []
        for epoch in range(epochs):
            epoch_loss = []
            for batch_idx, (data, target) in enumerate(trn_dataloader):
                data = torch.autograd.Variable(data)
                optimizer.zero_grad()
                pred = self.forward(data)
                loss = loss_func(pred, data)
                epoch_loss.append(loss.cpu().data.item())
                loss.backward()
                optimizer.step()
                
            epoch_loss = np.array(epoch_loss)
            mean_loss = np.mean(epoch_loss)
            print('Train epoch: {}, loss: {}'.format(epoch, mean_loss))
            losses.append(mean_loss)    
        return np.array(losses)
        
    
    def get_encoding(self, train_data):
        x_train = Variable(torch.from_numpy(train_data)).type(torch.FloatTensor)
        y_train = Variable(torch.from_numpy(train_data)).type(torch.FloatTensor)
        
        trn = TensorDataset(x_train, y_train)
        trn_dataloader = torch.utils.data.DataLoader(trn, batch_size=1, shuffle=False, num_workers=2)
        
        #print(self)
        encoded_data = []
        
        for batch_idx, (data, target) in enumerate(trn_dataloader):
            data = torch.autograd.Variable(data)
            enc = self.encode(data).detach().numpy()
            enc = enc.flatten()
            encoded_data.append(enc)
            
        return np.array(encoded_data)
        
            
    
class Stacked_AE:
    def __init__(self, N_layers, dims):
        """
        Parameters
        ----------
        N_layers: Number of AE to stack
        dims : dimension of input followed by 
                hidden layer (array, length = N_layers + 1)

        Returns
        -------
        None.
        """
        if(len(dims)!=N_layers+1):
            print("Entered wrong dimensions, dimension of input followed by hidden layers")
            exit(0);
        self.AANNs = []
        self.N_layers = N_layers
        self.dims = dims
        for i in range(N_layers):
            model = AE(dims[i], dims[i+1])
            self.AANNs.append(model)
    
    
    def encoding(self, i, data_set):
        return self.AANNs[i].get_encoding(data_set)
                                    
    
    def stack_training(self, train_data, epochs, learning_rate, batch_size):
        data_set = train_data;
        losses = []
        for i in range(self.N_layers):
            loss = self.AANNs[i].train(data_set, epochs, learning_rate, batch_size)
            losses.append([loss])
            data_set = self.encoding(i, data_set)
        return losses
            

class MLFFNN(nn.Module):
    def __init__(self, dims):
        """
        Parameters
        ----------
        dims : array of size of all layers
        """
        super(MLFFNN, self).__init__()
        self.dims = dims
        self.n_layers = len(dims)
        self.layer = nn.ModuleList()
        for i in range(1, len(dims)):
            self.layer.append(nn.Linear(dims[i-1], dims[i]))

    def forward(self, x):
        
        for i in range(self.n_layers - 2):
            x = self.layer[i](x)
            x = F.relu(x)
        x = self.layer[self.n_layers - 2](x)
        return x

    def train(self, train_data, train_results, epochs, learning_rate, batch_size):
        x_train = Variable(torch.from_numpy(train_data)).type(torch.FloatTensor)
        y_train = Variable(torch.from_numpy(train_results))
        
        trn = TensorDataset(x_train, y_train)
        trn_dataloader = torch.utils.data.DataLoader(trn, batch_size=batch_size, shuffle=True, num_workers=2)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        print(self)
        
        losses = []
        for epoch in range(epochs):
            epoch_loss = []
            for batch_idx, (data, label) in enumerate(trn_dataloader):
                data = torch.autograd.Variable(data)
                optimizer.zero_grad()
                pred = self.forward(data)
                loss = F.cross_entropy(pred, label)
                epoch_loss.append(loss.cpu().data.item())
                loss.backward()
                optimizer.step()
                
            
            epoch_loss = np.array(epoch_loss)
            mean_loss =  np.mean(epoch_loss)
            print('Train epoch: {}, loss: {}'.format(epoch, mean_loss))
            losses.append(mean_loss)
        return losses
    
    def get_class(self, dataset, true_class):
        with torch.no_grad():
            x_inp = Variable(torch.from_numpy(dataset)).type(torch.FloatTensor)
            y_inp = Variable(torch.from_numpy(true_class))
            trn = TensorDataset(x_inp, y_inp)
            dataloader = torch.utils.data.DataLoader(trn, batch_size=1, shuffle=False)
            preds = []
            for batch_idx, (data, target) in enumerate(dataloader):
                pred = self.forward(data).detach().numpy()
                preds.append(np.argmax(pred))
        return preds
