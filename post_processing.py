#!/usr/bin/env python
# coding: utf-8

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
import torch
import random
import time
import pickle
import csv


## 1. first step of post_processing: correct pointwsie predictions
def count_modes(y_arr,k=5):
    cnt = [0 for _ in range(k)]
    for i in range(k):
        cnt[i] = np.sum(y_arr == i)
    cnt = cnt / np.sum(cnt)
    return cnt

def post_process_trip_middle(y,scale,thd,loop=False):
    # length of y must be greater than 2*scale
    length = len(y)
    assert length>2*scale
    y_old = y.copy()
    y_new = y.copy()
    while True:
        had_change = False
        for i in range(scale,length-scale):
            cnt_mode = count_modes(y_old[(i-scale):(i+scale+1)])
            label = np.where(cnt_mode >= thd)[0]
            if len(label)>0 and y_new[i]!=label[0]:
                y_new[i] = label[0]
                had_change = True
        if not loop or not had_change:
            break
        y_old = y_new.copy()
    return y_new

def post_process_trip_end(y,scale,thd,scale_end,loop=False):
    # length of y must be greater than 2*scale    
    y_new = y.copy()
    num = len(y)
    y_new[0:(scale + scale_end)] = post_process_trip_middle(y[0:(scale+scale_end)],scale_end,thd,loop)
    y_new[(num-scale-scale_end):] = post_process_trip_middle(y[(num-scale-scale_end):],scale_end,thd,loop)
    if True:
        cnt = count_modes(y_new[0:scale_end])
        y_new[0:scale_end] = np.argmax(cnt)
        cnt = count_modes(y_new[(num-scale_end):])
        y_new[(num-scale_end):] = np.argmax(cnt)
    return y_new

def post_process_trip(y,scale,minlen,thd,scale_end=4,loop=False,deal_end=False):
    length = len(y)
    if len(np.unique(y)) == 1:
        return y
    if length <= minlen:
        cnt_mode = count_modes(y)
        label = np.argmax(cnt_mode)
        y = np.ones(length) * label
        return y.astype(np.int)
    elif length<2*scale+1:
        if deal_end:
            y = post_process_trip_end(y,scale,thd,loop=loop,scale_end=scale_end)
        return y
    y = post_process_trip_middle(y,scale,thd,loop)
    if deal_end:
        y = post_process_trip_end(y,scale,thd,loop=loop,scale_end=scale_end)
    return y

def post_process(y,scale=19,minlen=20,thd=0.7,k=5,scale_end=None,**kwargs):
    if 'deal_end' in kwargs.keys():
        deal_end = kwargs['deal_end']
    else:
        deal_end = False
    if 'loop' in kwargs.keys():
        loop = kwargs['loop']
    else:
        loop = False
    if scale_end is None:
        scale_end = 4
    y = post_process_trip(y,scale=scale,minlen=minlen,thd=thd,
                          scale_end=scale_end,loop=loop,deal_end=deal_end)
    return y

class PostProcesser:
    def __init__(self,scale=9,minlen=20,thd=0.7,k=5,**kwargs):
        self.scale = scale
        self.minlen = minlen
        self.thd = thd
        self.k = k
        if 'deal_end' in kwargs.keys():
            self.deal_end = kwargs['deal_end']
        else:
            self.deal_end = False
        if 'loop' in kwargs.keys():
            self.loop = kwargs['loop']
        else:
            self.loop = False
        if 'scale_end' in kwargs.keys():
            self.scale_end = kwargs['scale_end']
        else:
            self.scale_end = None
        if 'save_path' in kwargs.keys():
            self.save_path = kwargs['save_path']
        else:
            self.save_path = None
            
    def __call__(self,y,masking=None):
        '''
        Data type should be numpy array.
        :param y : 1-d array , prediction of a batch data;
        :param masking : 2-d array ,[B,L], masking of real data length.
        '''
        if isinstance(y,torch.Tensor):
            y = y.numpy()
        if isinstance(masking,torch.Tensor):
            masking = masking.numpy()
            
        if masking is None:
            return post_process(y,scale=self.scale,minlen=self.minlen,scale_end = self.scale_end,
                                thd=self.thd,k=self.k,deal_end=self.deal_end,loop=self.loop)
        # when received a batchsize 
        # save result
        if self.save_path is not None:
            self.save_result(self.save_path,[y,masking])
        if len(masking.shape)==2:
            num_list = [np.sum(masking[i]) for i in range(masking.shape[0])]
        else:
            num_list = [np.sum(masking)]
        srt = 0
        y_post = np.zeros(np.sum(num_list),dtype=np.int)
        for num in num_list:
            y_post[srt:(srt+num)] = post_process(y[srt:(srt+num)],scale=self.scale,minlen=self.minlen,
                                                 scale_end = self.scale_end,thd=self.thd,k=self.k,
                                                 deal_end=self.deal_end,loop=self.loop)
            srt = srt + num
        return y_post
    
    def save_result(self,save_path,data):
        with open(save_path,'wb') as f:
            pickle.dump(data,f)


## 2. second step of post_processing: correct segment prediciton
def find_change_point(y_trip):
    index_change = [0]
    if len(np.unique(y_trip)) == 1:
        index_change.append(len(y_trip))
        return index_change
    for i in range(1,len(y_trip)):
        if y_trip[i] == y_trip[i-1]:
            continue
        index_change.append(i)
    index_change.append(len(y_trip))
    return index_change

def merge_segment(y_trip,minlen_1=20,minlen_2=40):
    if len(np.unique(y_trip)) == 1:
        return y_trip
    index_change = find_change_point(y_trip)
    num = len(index_change)
    if  num < 4:
        return y_trip
    y_new = y_trip.copy()
    length_list = [index_change[i+1]-index_change[i] for i in range(num-1)]
    srt = 0
    for i,length in enumerate(length_list):
        if length<minlen_1 and i>0 and i<num-2:
            if length_list[i-1]>minlen_2 and length_list[i+1]>minlen_2:
                tmp = np.random.randint(0,2,1)
                if tmp == 0:
                    y_new[srt:(srt+length)] = y_trip[srt-1]
                else:
                    y_new[srt:(srt+length)] = y_trip[srt+length]
        srt = srt + length
    return y_new

def merge_all(y,masking=None,minlen_1=20,minlen_2=40):
    if masking is None:
        y = merge_segment(y,minlen_1,minlen_2)
        return y
    num_list = [np.sum(masking[i]) for i in range(masking.shape[0])]
    srt = 0
    for num in num_list:
        y[srt:(srt+num)] = merge_segment(y[srt:(srt+num)],minlen_1,minlen_2)
        srt = srt + num
    return y

class SegmentMerge:
    def __init__(self,minlen_1=20,minlen_2=40,**kwargs):
        self.minlen_1 = minlen_1
        self.minlen_2 = minlen_2
    
    def __call__(self,y,masking=None):
        return merge_all(y,masking,minlen_1=self.minlen_1,minlen_2=self.minlen_2)
        
        
## 3. Visualize results of post-processing
def visual_result(y_true,y_pred=None,y_post=None,y_merge=None,s=5,**kwargs):
    def plot_text():
        plt.text(0,0.5,'true')
        if y_pred is not None:
            plt.text(0,1.5,'pred')
        if y_post is not None:
            plt.text(0,2.5,'post')
        if y_merge is not None:
            plt.text(0,3.5,'merge')
    def plot_text_ch():
        plt.text(0,0.5,'真实类别')
        if y_pred is not None:
            plt.text(0,1.5,'预测类别')
        if y_post is not None:
            plt.text(0,2.5,'第一步后处理')
        if y_merge is not None:
            plt.text(0,3.5,'第二步后处理')
            
    # y_post must be None if y_pred is None
    colors = {0:'red',1:'black',2:'blue',3:'green',4:'orange'}
    if 'is_CH' in kwargs.keys():
        is_CH = kwargs['is_CH']
    else:
        is_CH = False
    if is_CH is False:
        legend = ['walk','bike','bus','car','train']
    else:
        legend = ['步行','自行车','公交车','小汽车','列车']
        
    num = len(y_true)
    if y_pred is None:
        y_new = np.zeros((num,1))
        y_new = y_true
    elif y_post is None:
        y_new = np.zeros((num,2))
        y_new[:,0] = y_true
        y_new[:,1] = y_pred
    elif y_merge is None:
        y_new = np.zeros((num,3))
        y_new[:,0] = y_true
        y_new[:,1] = y_pred
        y_new[:,2] = y_post
    else:
        y_new = np.zeros((num,4))
        y_new[:,0] = y_true
        y_new[:,1] = y_pred
        y_new[:,2] = y_post
        y_new[:,3] = y_merge
    # size of figure
    if 'figsize' in kwargs.keys():
        plt.figure(figsize=kwargs['figsize'])
    for i in range(5):
        x = []
        y = np.zeros(0)
        for j in range(y_new.shape[1]):
            index = np.where(y_new[:,j] == i)[0]
            x.extend(index)
            y = np.append(y,np.ones(len(index))*j)
        plt.scatter(x,y,s=s,c=colors[i],label=legend[i])
    plt.ylim(-0.1,y_new.shape[1]+0.5)
    if is_CH is False:
        plot_text()
        plt.legend()
        plt.show()
    else:
        plot_text_ch()
        plt.xlabel('采样点序号')
        plt.legend()
        plt.show()
        

## 4. An example of post-processing using simulated data
def get_sim_data(num_list=[100],noise_ratio=0.1,noise=1,dtype='numpy'):
    num = np.sum(num_list)
    if dtype == 'numpy':
        y = np.zeros(num,dtype=np.int)
        srt = 0
        for i in range(len(num_list)):
            y[srt:(srt+num_list[i])] = i
            srt = srt + num_list[i]
        num_noise = np.floor(num*noise_ratio).astype(np.int)
        index_noise = random.sample(range(num),num_noise)
        y_pred = y.copy()
        y_pred[index_noise] = noise
    elif dtype == 'torch':
        y = torch.zeros(num)
        srt = 0
        for i in range(len(num_list)):
            y[srt:(srt+num_list[i])] = i
            srt = srt + num_list[i]
        num_noise = np.floor(num*noise_ratio).astype(np.int)
        index_noise = random.sample(range(num),num_noise)
        y_pred = y.clone()
        y_pred[index_noise] = noise
    else:
        raise ValueError('Unsupported data type:',dtype)
    return y_pred,y

def accuracy(y_pred,y):
    if isinstance(y,torch.Tensor):
        acc = torch.sum(y_pred == y).item() / torch.tensor(len(y)).item()
    else:
        acc = np.sum(y_pred == y) / len(y)
    return acc

def test_post_process(y_pred,y,scale=9,minlen=20,thd=0.8,s=5,figsize=(10,5)):
    acc = accuracy(y_pred,y)
    print('without post processing:acc =',acc)
    y_post = post_process(y_pred,scale=scale,minlen=minlen,thd=thd)
    acc = accuracy(y_post,y)
    print('after post processing:acc =',acc)
    visual_result(y,y_pred,y_post,s=s,figsize=figsize)
    
if __name__ == '__main__':
    y_pred,y = get_sim_data(num_list=[100],noise_ratio=0.2,noise=2,dtype='numpy')
    test_post_process(y_pred,y,scale=9,minlen=20,thd=0.6,s=2,figsize=(10,5))