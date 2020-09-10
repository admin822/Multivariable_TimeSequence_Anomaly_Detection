import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.append(BASE_DIR)
from models.RNN_VAE import rnn_vae
from datetime import datetime
import time
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn
import copy
import pickle
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from training.training_visulization import visualize_with_visdom
from visdom import Visdom
import math
os.environ['KMP_DUPLICATE_LIB_OK']='True'
LOG_2_PI=math.log(math.pi)




def rnn_vae_fit(training_set:DataLoader,validation_set:DataLoader,device:torch.device,save_model_path:str,save_log_path:str,num_epochs=200,lr=1e-5,visulization=False,patience=10):
    model=rnn_vae()
    model=nn.DataParallel(model)
    model=model.to(device)
    optm=torch.optim.Adam(model.parameters(),lr=lr)
    best_loss=float('inf')
    best_model=copy.deepcopy(model.state_dict())
    counter_no_improve=0
    if(save_log_path!=None):
        epochs_so_far=[]
    training_losses_so_far=[]
    dev_losses_so_far=[]
    epochs=[]
    # this win_id is for visualization in visdom
    win_id=None
    for e in range(num_epochs):
        model.train()
        print("Current Epoch Number:{}".format(e+1))
        start_time=time.time()
        training_losses=[]
        dev_losses=[]
        counter=0
        print("\tTraining Starts:")
        for x in training_set:
            # x:{batch_size,seq_len,dim}
            x=x.float()
            x=x.to(device)
            to_be_tested=[]
            for seq in x:
                to_be_tested.append(seq[-1].view(1,seq.shape[1]))
            to_be_tested=torch.cat(to_be_tested,0)
            to_be_tested=to_be_tested.to(device)
            optm.zero_grad()
            z_mu,z_logcovar,post_mu,post_logcovar=model(x)
            # all the output has shape [batch_size,latent_space_dim]
            KL= -0.5 * torch.sum(1. + z_logcovar - z_mu.pow(2) -z_logcovar.exp()) / x.shape[0]
            negative_loglikelihodd=torch.sum(LOG_2_PI+post_logcovar+(to_be_tested-post_mu)**2/(2*torch.exp(post_logcovar)))/to_be_tested.shape[0]
            loss=negative_loglikelihodd+KL
            loss.backward()
            optm.step()
            training_losses.append(loss.item())
            counter+=1
            if(counter%200==1):
                print("\t\tIn batch {}, the training loss is {}".format(counter,loss.item()))
        model.eval()
        print("\tValidation Starts:")
        with torch.no_grad():
            counter=1
            for y in validation_set:
                y=y.float()
                y=y.to(device)
                to_be_tested=[]
                for seq in y:
                    to_be_tested.append(seq[-1].view(1,seq.shape[1]))
                to_be_tested=torch.cat(to_be_tested,0)
                to_be_tested=to_be_tested.to(device)
                z_mu,z_logcovar,post_mu,post_logcovar=model(y)
                # all the output has shape [batch_size,latent_space_dim]
                KL= -0.5 * torch.sum(1. + z_logcovar - z_mu.pow(2) -z_logcovar.exp()) / x.shape[0]
                negative_loglikelihodd=torch.sum(LOG_2_PI+post_logcovar+(to_be_tested-post_mu)**2/(2*torch.exp(post_logcovar)))/to_be_tested.shape[0]
                loss=negative_loglikelihodd+KL
                dev_losses.append(loss.item())
                if(counter%50==1):
                        print("\t\tIn batch {}, the validation loss is {}".format(counter,loss.item()))
                counter+=1
        train_loss_this_epoch=np.mean(training_losses)
        dev_loss_this_epoch=np.mean(dev_losses)
        print("Epoch:{}\tTrain_losss:{}\tValidation_loss:{}\tTime:{}".format(e+1,train_loss_this_epoch,dev_loss_this_epoch,round(time.time()-start_time,2)))
        if(save_log_path!=None):
            epochs_so_far.append((train_loss_this_epoch,dev_loss_this_epoch))
        epochs.append(e+1)
        training_losses_so_far.append(train_loss_this_epoch)
        dev_losses_so_far.append(dev_loss_this_epoch)
        if(visulization):
            return_id=visualize_with_visdom(epochs,training_losses_so_far,dev_losses_so_far,win_id)
            if(return_id!=None):
                win_id=return_id
        if(save_log_path!=None):
            with open(save_log_path,'wb') as f:
                    pickle.dump(epochs_so_far,f)
        if(dev_loss_this_epoch<best_loss):
            counter_no_improve=0
            best_loss=dev_loss_this_epoch
            best_weights=copy.deepcopy(model.state_dict())
            torch.save(best_weights,save_model_path)
        else:
            counter_no_improve+=1
            if(counter_no_improve>=patience):
                print("Met early stop, stopped at epoch {}".format(e+1))
                break






