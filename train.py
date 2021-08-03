#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import time
import pickle

import point_net_variant
import my_metrics




# # 1.Basic training functions

# ## 1.1 trainer class for deep learning
class DLTrainer:
    '''
    A class for training deep learning networks.
    '''
    def __init__(self,model,num_classes,lr=0.001,save_path='model.pt'):
        self.model = model
        self.lr = lr
        self.lr_list = []
        self.num_classes = num_classes
        self.save_path = save_path
        self.optimizer = torch.optim.Adam(model.parameters(),lr = lr)
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        # early stopping
        self.patience = None
        self.best_score = None
        self.early_stop_cnt = 0
        self.val_loss_min = np.Inf
    
    def format_second(self,seconds):
        if seconds < 1:
            return str(seconds)+"s"
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return '%02d:%02d:%02d'%(h,m,s)
    
    def validate_step(self,data):
        x,y = data
        pred = self.model(x)
        # create masking 
        masking = y >= 0
        pred_valid = pred[masking]
        y_valid = y[masking]
        loss = self.loss_func(pred_valid,y_valid)
        
        y_pred = pred_valid.max(1)[1]
        return y_pred,loss.item()
        
    def validate(self,data,batch_size):
        '''
        Evaluate performance of model on validation set.
        
        '''
        x,y = data
        n_samples = len(y)
        num_batch = int(np.ceil(n_samples/batch_size))
        y_pred = None
        val_loss = 0
        for k in range(num_batch):
            if k == num_batch-1:
                inx_end = min((k+1) * batch_size,n_samples)
            else:
                inx_end = (k+1) * batch_size
            inx_srt = k * batch_size
            data = [x[inx_srt:inx_end],y[inx_srt:inx_end]]
            y_pred_batch,batch_loss = self.validate_step(data)
            val_loss += batch_loss
            if y_pred is None:
                y_pred = y_pred_batch 
            else:
                y_pred = torch.cat([y_pred,y_pred_batch],dim=0)
        y = y[y>=0]
        self.val_loss.append(val_loss/len(y))
        self.val_acc.append(my_metrics.metrics(y_pred.cpu(),y.cpu()))
        
    def early_stop(self,val_loss,min_delta):
        score = - val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss)
        elif score < self.best_score + min_delta:
            self.early_stop_cnt += 1
            if self.early_stop_cnt >= self.patience:
                print("Training early stopping because val_loss didn't decrease ",min_delta,'for ',self.patience,'epochs.')
                return True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss,save_weights_only=self.save_weights_only)
            self.early_stop_cnt = 0
        return False
    
    def save_checkpoint(self,val_loss=None,save_weights_only=True):
        '''
        Save model when validation loss decrease.
        '''
        if val_loss is None:
            print('save model to ',self.save_path)
        else:
            print(f'Validation loss decreased ({self.val_loss_min:.6f}-->{val_loss:.6f},saving model to {self.save_path}')
            self.val_loss_min = val_loss
        if save_weights_only:
            torch.save(self.model.state_dict(),self.save_path)
        else:
            torch.save(self.model,self.save_path)
        
    def train_step(self,data):
        x,y = data
        self.optimizer.zero_grad()
        pred = self.model(x) # [B,k] for class,[B,n_pts,k] for seg
        # create masking 
        masking = y >= 0
        pred_valid = pred[masking]
        y_valid = y[masking]

        loss = self.loss_func(pred_valid,y_valid)
        loss.backward()
        self.optimizer.step()
        
        return loss.item(),y_valid,pred_valid.max(1)[1]
    
    def train(self,data,batch_size):
        X,Y = data
        n_samples = len(Y)
        num_batch = int(np.ceil(n_samples/batch_size))
        loss_epoch ,N_valid = 0,0
        Y_valid,Y_pred_valid = None,None
        for k in range(num_batch):
            if k == num_batch-1:
                inx_end = min((k+1)*batch_size,n_samples)
            else:
                inx_end = (k+1)*batch_size
            inx_srt = k*batch_size
            if inx_srt == inx_end - 1:
                continue    # drop out batch containing only one sample.
            y = Y[inx_srt:inx_end]
            x = X[inx_srt:inx_end]  # [B,n_pts,n_features]

            loss_batch,y_valid,y_pred_valid = self.train_step([x,y])
            loss_epoch += loss_batch
            N_valid += len(y_valid)
            if Y_valid is None:
                Y_valid = y_valid
                Y_pred_valid = y_pred_valid
            else:
                Y_valid = torch.cat((Y_valid,y_valid),dim=0)
                Y_pred_valid = torch.cat((Y_pred_valid,y_pred_valid),dim=0)
                
        # save train loss and acc
        self.loss.append(loss_epoch/N_valid)
        if self.use_gpu:
            Y_pred_valid = Y_pred_valid.cpu()
            Y_valid = Y_valid.cpu()
        self.acc.append(my_metrics.metrics(Y_pred_valid,Y_valid))
        
    def show_training_process(self,verbose=1,delta_time=None):
        if verbose != 1:
            return
        if self.patience is not None:
            print('[%d/%d] - loss:%f - acc:%f - val_loss:%f - val_acc:%f - %s'%(self.cur_epoch,self.epochs,self.loss[-1],self.acc[-1],
                   self.val_loss[-1],self.val_acc[-1],self.format_second(delta_time)))
        else:
             print('[%d/%d] loss:%f - acc:%f - %s'%(self.epoch,self.epochs,self.loss[-1],self.acc[-1],self.format_second(delta_time)))
    
    def scheduler_step(self):
        if self.scheduler is None:
            return
        self.scheduler.step()
        self.lr_list.append(self.scheduler.get_last_lr[0])
        
    def fit(self,X_train,Y_train,X_val=None,Y_val=None,
            batch_size = 32,
            n_epochs = 30,
            patience = None, 
            min_delta = 0.005 ,
            loss_func = torch.nn.NLLLoss(reduction='sum'),
            scheduler = None,
            verbose = 1,
            use_gpu = False,
            save_weights_only = True,
            **kwargs):
        '''
        Train network given parameters.
        :param X_train : tensor of size [samples,maxlen,n_features] or [samples,timesteps,features]
        :param Y_train : tensor of size [samples] or [samples,timesteps]
        :param patience :must be None or an positive interger,standing for 
            that if loss don't drop min_delta after early_stopping epochs ,then we
            stop training.If the param is not None,then X_val and Y_val must be not 
            None.
        
        '''
        if patience is not None and (X_val is None or Y_val is None):
            raise ValueError('When patience is not None,X_val and Y_val must be not None too.')
        self.batch_size = batch_size
        self.patience = patience
        self.loss_func = loss_func
        self.epochs = n_epochs
        self.scheduler = scheduler
        self.save_weights_only = save_weights_only
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_gpu else "cpu")
        
        start_time1 = time.process_time()
        self.model.to(self.device)
        X_train = X_train.to(self.device)
        Y_train = Y_train.to(self.device)
        for epoch in range(n_epochs):
            start_time2 = time.process_time()
            # set model to training mode
            self.cur_epoch = epoch + 1
            self.model.train()
            self.train([X_train,Y_train],self.batch_size)
            
            if patience is None:
                delta_time = time.process_time()-start_time2
                self.show_training_process(verbose=verbose,delta_time=delta_time)
                continue
            # set model to evaluation mode
            if self.use_gpu:
                X_val = X_val.to(self.device)
                Y_val = Y_val.to(self.device)
            self.model.eval()
            self.validate([X_val,Y_val],self.batch_size)
            
            # schedule learing rate
            self.scheduler_step()
            
            delta_time = time.process_time()-start_time2
            self.show_training_process(verbose=verbose,delta_time=delta_time)
            
            if self.early_stop(self.val_loss[-1],min_delta):
                break
        print('%d epochs for training finished - %s'%
              (n_epochs,self.format_second(time.process_time()-start_time1)))
        # save model after all epochs if didn't set early stopping.
        if patience is None:
            self.save_checkpoint(save_weights_only = self.save_weights_only)


# ## 1.2 lr functions 
def get_scheduler(optimizer,policy='exp',**kwargs):
    '''Return lr scheduler'''
    if policy == 'step':
        gamma = kwargs['gamma'] if 'gamma' in kwargs else 0.1
        scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size = kwargs['step_size'],
                    gamma = gamma)
    elif policy == 'plateau': 
        factor = kwargs['factor'] if 'factor' in kwargs else 0.1
        patience = kwargs['patience'] if 'patience' in kwargs else 10
        min_lr = kwargs['min_lr'] if 'min_lr' in kwargs else 0
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode = 'min',
                    factor = factor,
                    #threshold = kwargs['threshold'],
                    patience = patience)
    elif policy == 'exp':
        gamma = kwargs['gamma'] if 'gamma' in kwargs else 0.1
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer,
                    gamma = gamma
                    )
    elif policy == 'cosine':
        eta_min = kwargs['eta'] if 'eta' in kwargs else 0
        T_max = kwargs['T_max'] if 'T_max' in kwargs else 10
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max = T_max,
                    eta_min = eta_min # minimum learning rate
                    )
    else:
        raise ValueError('Unsupported scheduler policy:',policy)
    return scheduler


# # 2. Train PointNet(vanilla)

# ## 2.1 3P-MSPointNet for semantic segmentation
'''
Load training or testing data and transform them into torch tensors.
NOTE: this requires that data have been split into training, validaiton
    and test set, and stored as [X_train,X_val,X_test,Y_train,Y_val,Y_test] 
    in `pickle` format. Among them, X with shape of (samples,timesteps,n_features)
    and Y with shape (samples,timesteps). Please change the funtion if your data 
    is not stored like this.
'''
def load_data(data_path,is_train=True,task='cls'):
    if is_train:
        with open(data_path,'rb') as f:
            if task == 'cls':
                X_train,X_val,_,Y_train,Y_val,_,_,_,_ = pickle.load(f)
            elif task == 'seg':
                X_train,X_val,_,Y_train,Y_val,_ = pickle.load(f)
        print('train of X,Y:',X_train.shape,Y_train.shape)
        print('val of X,Y:',X_val.shape,Y_val.shape)
        X_train = Variable(torch.from_numpy(X_train).float())
        X_val = Variable(torch.from_numpy(X_val).float())
        Y_train = Variable(torch.from_numpy(Y_train).type(torch.long))
        Y_val = Variable(torch.from_numpy(Y_val).type(torch.long))
        return X_train,X_val,Y_train,Y_val
    else:
        with open(data_path,'rb') as f:
            if task == 'cls':
                _,_,X_test,_,_,Y_test,_,_,_ = pickle.load(f)
            elif task == 'seg':
                _,_,X_test,_,_,Y_test = pickle.load(f)
        print('test of X,Y:',X_test.shape,Y_test.shape)
        X_test = Variable(torch.from_numpy(X_test).float())
        Y_test = Variable(torch.form_numpy(Y_test).float())
        return X_test,Y_test



# ## 2.2 3P-MSPointNetfor segmentation 
if __name__ == '__main__':
    num_classes = 5
    lr = 0.001
    save_path = './weights/seg_PointNet_Vanilla.pt'
    model = point_net_variant.MSPointNetDenseCls_V3(k=num_classes)
    trainer = DLTrainer(model,num_classes,lr = lr,save_path=save_path)
    # set params for training
    batch_size = 64
    n_epochs = 50
    patience = 7
    # load your training data set, change this if necessary
    data_path = './data/trips_fixed_len_400.pickle'
    X_train,X_val,Y_train,Y_val = load_data(data_path,is_train=True,task='seg')
    # start training
    trainer.fit(X_train,Y_train,X_val,Y_val,
                  batch_size = batch_size,
                  n_epochs=n_epochs,
                  patience=patience)



# visualize the training process
# plt.plot(range(len(trainer.loss)),trainer.loss,'r')
# plt.plot(range(len(trainer.val_loss)),trainer.val_loss,'g')



