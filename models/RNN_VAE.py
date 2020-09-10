from torch import nn
import torch.nn.functional as F
from torch import optim
import torch
import random
import time
import os

## This will be a model that implements RNN and variational autoencoder to accomplish the task of multivariable time-series analysis, model will also be able to detect which parameter is anomalous


## class filter_layer()
# this will further serve as a 'filter' that seperates noise from no-noise data in the original dataset.



class encoder(nn.Module):
    def __init__(self):
        super().__init__()
        #should be able to change to different kinds of rnn,input_size,hidden_size,layers and stuff.
        self.rnn=nn.LSTM(371,180,batch_first=True)
        self.act1=nn.Softplus()
        self.mu=nn.Linear(180,90)
        self.logcovar=nn.Linear(180,90)
    def forward(self,x):
        _,(output,_)=self.rnn(x)
        output.squeeze_(0)
        output=self.act1(output)
        mu=self.mu(output)
        logcovar=self.logcovar(output)
        return mu,logcovar

class decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.post_mu=nn.Sequential(
        nn.Linear(90,180),
        nn.BatchNorm1d(180),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(180,371)
        )
        self.post_logcovar=nn.Sequential(
            nn.Linear(90,180),
            nn.BatchNorm1d(180),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(180,371),
            nn.Tanh()
        )
    def forward(self,z):
        post_mu=self.post_mu(z)
        post_logcovar=self.post_logcovar(z)
        return post_mu,post_logcovar
        

class rnn_vae(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=encoder()
        self.decoder=decoder()
    def forward(self,x):
        z_mu,z_logcovar=self.encoder(x)
        z_std=torch.exp(z_logcovar/2)
        e=torch.randn_like(z_mu)*z_std+z_mu
        post_mu,post_logcovar=self.decoder(e)
        return z_mu,z_logcovar,post_mu,post_logcovar


