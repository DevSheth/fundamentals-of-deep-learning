import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class RBM():
    def __init__(self, d, l, DEVICE, real=True, gaussian=False, var=None):
        self.d = d
        self.l = l
        self.w = torch.randn(l,d).to(DEVICE)
        self.b = torch.randn(1,d).to(DEVICE)
        self.c = torch.randn(1,l).to(DEVICE)
        self.real = real
        self.DEVICE = DEVICE
        self.gaussian = gaussian
        if var is not None:
            self.var = var.to(DEVICE)
        self.Dw = torch.zeros(l,d).to(DEVICE)
        self.Db = torch.zeros(1,d).to(DEVICE)
        self.Dc = torch.zeros(1,l).to(DEVICE)
        self.Qw = torch.zeros(l,d).to(DEVICE)
        self.Qb = torch.zeros(1,d).to(DEVICE)
        self.Qc = torch.zeros(1,l).to(DEVICE)
        self.Rw = torch.zeros(l,d).to(DEVICE)
        self.Rb = torch.zeros(1,d).to(DEVICE)
        self.Rc = torch.zeros(1,l).to(DEVICE)
        self.ad_cnt = 0
        
    def sample_hidden(self, x):
        a = torch.add(torch.matmul(x, torch.t(self.w)), self.c)
        p_h_v = torch.sigmoid(a)
        h = torch.bernoulli(p_h_v)
        return p_h_v, h
    
    def sample_hidden_gaussian(self, x):
        a = torch.add(torch.matmul(x/self.var, torch.t(self.w)), self.c)
        p_h_v = torch.sigmoid(a)
        h = torch.bernoulli(p_h_v)
        return p_h_v, h
    
    def sample_visible(self, h):
        a = torch.add(torch.matmul(h, self.w), self.b)
        p_v_h = torch.sigmoid(a)
        v = torch.bernoulli(p_v_h)
        return p_v_h, v
    
    def sample_visible_gaussian(self, h):
        mean = torch.add(torch.matmul(h, self.w), self.b)
        p_v_h = torch.distributions.normal.Normal(mean, self.var).sample()
        return p_v_h
    
    def get_output(self, x):
        if(not(self.gaussian)):
            p_h_v, h = self.sample_hidden(x)
            p_v_h, v = self.sample_visible(h)
            if(self.real):
                return p_v_h
            else:
                return v
        else:
            p_h_v, h = self.sample_hidden_gaussian(x)
            v = self.sample_visible_gaussian(h)
            return v
    
    def contrastive_divergence(self, k, x, optim="Delta", lr=0.01, alpha=0.9, rho=(0.9, 0.999)):
        if(not(self.gaussian)):
            v_0 = x.clone().detach()
            p_h_v_0, _ = self.sample_hidden(x)
            v = x.clone().detach()
            
            if(self.real):
                for i in range(k):
                    p_h_v, h = self.sample_hidden(v)
                    v, _ = self.sample_visible(h)
                v_k = v.clone().detach()
                p_h_v_k, _ = self.sample_hidden(v)
            else:
                for i in range(k):
                    p_h_v, h = self.sample_hidden(v)
                    p_v_h, v = self.sample_visible(h)
                v_k = v.clone().detach()
                p_h_v_k, _ = self.sample_hidden(v)
            
            del_w = torch.zeros(self.l,self.d).to(self.DEVICE)
            del_b = torch.zeros(1,self.d).to(self.DEVICE)
            del_c = torch.zeros(1,self.l).to(self.DEVICE)
            bs = x.shape[0]
            for i in range(bs):
                del_w_1 = torch.mm(p_h_v_0[i].t(), v_0[i])
                del_w_2 = torch.mm(p_h_v_k[i].t(), v_k[i])
                del_w += (del_w_1 - del_w_2)
                del_b += (v_0[i] - v_k[i])
                del_c += (p_h_v_0[i] - p_h_v_k[i])
        else:
            v_0 = x.clone().detach()
            p_h_v_0, _ = self.sample_hidden_gaussian(x)
            v = x.clone().detach()
            
            for i in range(k):
                p_h_v, h = self.sample_hidden_gaussian(v)
                v = self.sample_visible_gaussian(h)
            v_k = v.clone().detach()
            p_h_v_k, _ = self.sample_hidden_gaussian(v)
            
            del_w = torch.zeros(self.l,self.d).to(self.DEVICE)
            del_b = torch.zeros(1,self.d).to(self.DEVICE)
            del_c = torch.zeros(1,self.l).to(self.DEVICE)
            bs = x.shape[0]
            for i in range(bs):
                del_w_1 = torch.mm(p_h_v_0[i].t(), v_0[i]/self.var)
                del_w_2 = torch.mm(p_h_v_k[i].t(), v_k[i]/self.var)
                del_w += (del_w_1 - del_w_2)
                del_b += (v_0[i] - v_k[i])/self.var
                del_c += (p_h_v_0[i] - p_h_v_k[i])
        
        if(optim == "Delta"):
            self.Delta(lr, del_w/bs, del_b/bs, del_c/bs)
        elif(optim == "GenDelta"):
            self.GenDelta(lr, alpha, del_w/bs, del_b/bs, del_c/bs)
        elif(optim == "Adam"):
            self.Adam(lr, rho, del_w/bs, del_b/bs, del_c/bs)
        
        return v_k
    
    def Delta(self, lr, del_w, del_b, del_c):
        self.w += lr*del_w
        self.b += lr*del_b
        self.c += lr*del_c
        
    def GenDelta(self, lr, alpha, del_w, del_b, del_c):
        self.Dw = lr*del_w + alpha*self.Dw
        self.Db = lr*del_b + alpha*self.Db
        self.Dc = lr*del_c + alpha*self.Dc
        self.w += self.Dw
        self.b += self.Db
        self.c += self.Dc
    
    def Adam(self, lr, rho, del_w, del_b, del_c):
        self.ad_cnt += 1
        
        self.Qw = rho[0]*self.Qw + (1-rho[0])*del_w
        Qw_hat = self.Qw/(1 - rho[0]**self.ad_cnt)
        self.Rw = rho[1]*self.Rw + (1-rho[1])*del_w*del_w
        Rw_hat = torch.sqrt(self.Rw/(1 - rho[1]**self.ad_cnt))
        
        self.Qb = rho[0]*self.Qb + (1-rho[0])*del_b
        Qb_hat = self.Qb/(1 - rho[0]**self.ad_cnt)
        self.Rb = rho[1]*self.Rb + (1-rho[1])*del_b*del_b
        Rb_hat = torch.sqrt(self.Rb/(1 - rho[1]**self.ad_cnt))
        
        self.Qc = rho[0]*self.Qc + (1-rho[0])*del_c
        Qc_hat = self.Qc/(1 - rho[0]**self.ad_cnt)
        self.Rc = rho[1]*self.Rc + (1-rho[1])*del_c*del_c
        Rc_hat = torch.sqrt(self.Rc/(1 - rho[1]**self.ad_cnt))
        
        self.w += lr*(Qw_hat/(1e-8 + Rw_hat))
        self.b += lr*(Qb_hat/(1e-8 + Rb_hat))
        self.c += lr*(Qc_hat/(1e-8 + Rc_hat))
        
    def train(self, dataset, k, EPOCH, BATCH_SIZE, LR=1e-3):
        mse = nn.MSELoss()
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                                shuffle=True, num_workers=0)
        loss_plot = []
        for epoch in range(EPOCH):
            running_loss = 0.0
            for i, data in enumerate(dataloader, 0):
                image = data.to(self.DEVICE)
                
                self.contrastive_divergence(k, image, optim="Adam", lr=LR)
                
                output = self.get_output(image)
                loss = mse(image, output)
                running_loss += loss.item()
                
                pr = 10
                if i % pr == pr-1:
                    print("[%d, %5d] loss: %.7f" % (epoch+1, i+1, running_loss/(i+1)))
                    # running_loss = 0.0
                    
            loss_plot.append(running_loss/(i+1))
        print("Finished Training")
        reps = torch.zeros(len(dataset), self.l)
        dataloader = DataLoader(dataset, batch_size=1,
                                shuffle=False, num_workers=0)
        for i, data in enumerate(dataloader, 0):
            image = data.to(self.DEVICE)
            _, reps[i] = self.sample_hidden(image)
        return reps, loss_plot
        

class stacked_RBM(nn.Module):
    def __init__(self, dims, weights=None, bias=None):
        super(stacked_RBM, self).__init__()
        self.fc1 = nn.Linear(dims[0], dims[1])
        self.fc2 = nn.Linear(dims[1], dims[2])
        self.fc3 = nn.Linear(dims[2], dims[3])
        self.fc4 = nn.Linear(dims[3], dims[4])
        if weights is not None:
            self.fc1.weight = nn.Parameter(weights[0], requires_grad=True)
            self.fc1.bias = nn.Parameter(bias[0], requires_grad=True)
            self.fc2.weight = nn.Parameter(weights[1], requires_grad=True)
            self.fc2.bias = nn.Parameter(bias[1], requires_grad=True)
            self.fc3.weight = nn.Parameter(weights[2], requires_grad=True)
            self.fc3.bias = nn.Parameter(bias[2], requires_grad=True)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x