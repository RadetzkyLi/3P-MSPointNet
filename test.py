#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import time
import pickle
import csv

import point_net_variant
import my_metrics


## 1. basic functions
#   1.1 save and print metrics
def save_metrics(acc,f1_avg,precision,recall,f1_score,cm,k,output_path):
    '''
    Save them as a csv for making confusion matrix in Word conviniently.
    '''
    columns = ['walk','bike','bus','car','train','recall','samples']
    table = np.zeros((k+2,k+2))
    table[0:k,0:k] = cm
    table[0:k,k] = recall
    table[0:k,k+1] = np.sum(cm,axis=1)
    table[k,0:k] = precision
    table[k+1,0:k] = f1_score
    table[k,k+1] = acc
    table[k+1,k+1] = f1_avg
    with open(output_path,'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(table)
     
def print_metrics(y_pred,y_true,cm_path=None,**kwargs):
    '''
    Print all kinds of cretirion of classification.
    :cm_path : path for saving comfusion matrix;
    '''
    acc = my_metrics.metrics(y_pred,y_true,'accuracy')
    precision = my_metrics.metrics(y_pred,y_true,'precision')
    recall = my_metrics.metrics(y_pred,y_true,'recall')
    f1_score = my_metrics.metrics(y_pred,y_true,'f1_score')
    f1_score_avg = my_metrics.metrics(y_pred,y_true,'f1_score','weighted')
    cm = my_metrics.metrics(y_pred,y_true,'confusion_matrix')
    print('   accuracy:',acc)
    print('weighted-f1:',f1_score_avg)
    print('  precision:',precision)
    print('     recall:',recall)
    print('   f1-score:',f1_score)
    print('confusion matrix:\n',cm)
    if cm_path is not None:
        save_metrics(acc,f1_score_avg,precision,recall,f1_score,cm,len(recall),cm_path)
        

#   1.2 functions for test performance of model
def test_step(model,data):
    x,y = data
    pred = model(x)
    # create masking 
    masking = y >= 0
    pred_valid = pred[masking]
    y_valid = y[masking]
    y_pred = pred_valid.max(1)[1]
    return y_pred


def test_model(X_test,Y_test,
                model,
                batch_size=16,
                num_classes=5,
                cm_path = None,
                use_gpu = False,
                **kwargs):
    '''
    Evaluate performance of model on test set. 
    :param X_test : torch tensor of size [n_smaples,timesteps,n_features];
    :param Y_test : torch tensor of size [n_smaples,timesteps];
    :param model : deep learning model that will be evaluated;
    :param batch_size : batch size for testing;
    :param num_classes : number of classes of transportation modes.
    :param post_process : You can pass a `post_process` function to execute additional 
        operations on predictions such as saving prediction results. This fucntion 
        receive two params, `Y_pred` of shape (n_pts,k) and `making` of shape (n_samples).
    '''
    def empty_cache():
        torch.cuda.empty_cache()
        
    if 'post_process' in kwargs.keys():
        post_process = kwargs['post_process']
    else:
        post_process = None
        
    n_samples = len(Y_test)
    num_batch = int(np.ceil(n_samples/batch_size))
    y_pred = None
    # get prediction of all test samples
    start_time = time.process_time()
    model.eval()
    X_test = Variable(torch.from_numpy(X_test).float())
    Y_test = Variable(torch.from_numpy(Y_test).float())
    if use_gpu and torch.cuda.is_available():
        model = model.cuda()
        X_test = X_test.cuda()
        Y_test = Y_test.cuda()
    for k in range(num_batch):
        if k == num_batch-1:
            inx_end = min((k+1) * batch_size,n_samples)
        else:
            inx_end = (k+1) * batch_size
        inx_srt = k * batch_size
        x = X_test[inx_srt:inx_end]
        y = Y_test[inx_srt:inx_end]
        pred = test_step(model,[x,y])
        if y_pred is None:
            y_pred = pred
        else:
            y_pred = torch.cat((y_pred,pred),dim=0)
    masking = Y_test>=0
    y_true = Y_test[masking]
    print('total',n_samples,'samples are tested,consuming time:',time.process_time()-start_time,'second.')
    if post_process is not None:
        y_pred = post_process(y_pred.cpu(),masking.cpu())
    else:
        y_pred = y_pred.cpu()
    if cm_path is None:
        print_metrics(y_pred,y_true.cpu())
    else:
        print_metrics(y_pred,y_true.cpu(),save_cm=True,cm_path=cm_path)
    empty_cache()
    

if __name__ == "__main__":
    model_param_path = '/' # your model param path
    num_classes = 5
    model = point_net_variant.MSPointNetDenseCls_V3(k=num_classes)
    model.load_state_dict(torch.load(model_param_path))
    # load test dataset
    data_path = './data/trips_fixed_len_400.pickle'
    with open(data_path,'rb') as f:
        _,_,X_test,_,_,Y_test = pickle.load(f)
    test_model(X_test,Y_test,model,batch_size=32,num_classes=num_classes)
    
    