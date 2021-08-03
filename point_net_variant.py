#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
import torch.nn.functional as F


from new_layers import NewConv1d,NewMaxPool1d




# # Multi-Scale PointNet V3:conv & pointwise pyramid pooling

class MSPointNetFeat_V3(nn.Module):
    def __init__(self,
                 n_features = 4,
                 padding = 'same',
                 dims_up = [64,128],
                 dims_ms = [256],
                 scales_conv = [3],
                 scales_pool = [7,49]
                ):
        super(MSPointNetFeat_V3,self).__init__()
        self.padding = padding
        # raise dimension
        dims_conv = [n_features] + dims_up
        self.conv_embed = self.get_conv(dims_conv)
        self.bn_embed = self.get_bn(dims_up)
        # conv
        dims_conv = [dims_up[-1]] + dims_ms
        self.conv = self.get_conv(dims_conv,scales_conv)
        self.bn_conv = self.get_bn(dims_ms)
        # pyramid maxpool
        self.maxpool = self.get_maxpool(scales_pool)
        
        self.relu = nn.ReLU()
        
    def get_conv(self,dims,scales=None):
        conv = []
        if scales is None:
            for i in range(1,len(dims)):
                conv.append(nn.Conv1d(dims[i-1],dims[i],1))
        else:
            for i in range(1,len(dims)):
                conv.append(NewConv1d(dims[i-1],dims[i],scales[i-1],stride=1,padding=self.padding))
        return nn.ModuleList(conv)
            
    def get_bn(self,dims):
        bn = [nn.BatchNorm1d(dims[i]) for i in range(len(dims))]
        return nn.ModuleList(bn)
    
    def get_maxpool(self,scales):
        maxpool = []
        for i in range(len(scales)):
            maxpool.append(NewMaxPool1d(scales[i],stride=1,padding=self.padding))
        return nn.ModuleList(maxpool)
    
    def forward(self,x):
        n_pts = x.size()[2]
        for i in range(len(self.conv_embed)):
            x = self.relu(self.bn_embed[i](self.conv_embed[i](x)))
        for i in range(len(self.conv)):
            x = self.relu(self.bn_conv[i](self.conv[i](x)))
        point_feat = x
        local_feat = None
        for i in range(len(self.maxpool)):
            tmp = self.maxpool[i](x)
            if local_feat is None:
                local_feat = tmp
            else:
                local_feat = torch.cat([local_feat,tmp],1)
        # global features
        dim = x.size()[1]
        x = torch.max(x,2,keepdim=True)[0]
        x = x.reshape(-1,dim)
        x = x.view(-1,dim,1).repeat(1,1,n_pts)
        if local_feat is None:
            return torch.cat([point_feat,x],1)
        return torch.cat([point_feat,local_feat,x],1)


# In[21]:

'''
The semantic segmentation networks of 3P-MSPointNet.

Brief:
    receive data of shape (N,C) as input and output pointwise classification 
    results of shape (N,k), where N denotes number of samples, C denotes
    number of channels and k denotes number of categories.

Parameters:
    n_features - number of features;
    k - the number of categories;
    p_dropout - dropout probability in fully connected layers;
    padding - the padding strategy when conv or pooling, should be one of `same`,
        `valid` and `casual`, which has the same meaning as that in conv of tensorflow.
    dims_up - the number of layers and corresponding hidden units of the first MLP as 
        embedding layer, e.g., [64,128] represent two layers with hidden units 64 and 
        128 succsively;
    dims_ms - the number of filters in convolution layer;
    scales_conv - filter size in convolution layer;
    scales_pool - number of pooling windows and pooling sizes.
    
Returns:
    Pointwise classification results of shape (N,k).
'''
class MSPointNetDenseCls_V3(nn.Module):
    def __init__(self,
                  n_features = 4,
                  k = 5,
                  p_dropout = 0.5,
                  padding = 'same',
                  dims_up = [64,128],
                  dims_down = [256,128],
                  dims_ms = [256],
                  scales_conv = [3],
                  scales_pool = [7,49]
                 ):
        super(MSPointNetDenseCls_V3,self).__init__()
        self.k = k
        self.feat = MSPointNetFeat_V3(n_features,
                                         padding = padding,
                                         dims_up = dims_up,
                                         dims_ms = dims_ms,
                                         scales_conv = scales_conv,
                                         scales_pool = scales_pool
                                        )
        # fully connected layers
        dim = dims_ms[-1] * (len(scales_pool)+2)
        self.fc1 = nn.Conv1d(dim,dims_down[0],1)
        self.fc2 = nn.Conv1d(dims_down[0],dims_down[1],1)
        self.fc3 = nn.Conv1d(dims_down[1],k,1)
        self.dropout1 = nn.Dropout(p = p_dropout)
        self.dropout2 = nn.Dropout(p = p_dropout)
        self.bn1 = nn.BatchNorm1d(dims_down[0])
        self.bn2 = nn.BatchNorm1d(dims_down[1])
        self.relu = nn.ReLU()
        
    def forward(self,x):
        batch_size = x.shape[0]
        n_pts = x.shape[1]
        x = x.transpose(1,2)  # [B,n_features,n_pts]
        x = self.feat(x)   # [B,C,L]
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x) # [B,k,n_pts]
        x = x.transpose(1,2) #[B.n_pts,k]
        x = F.log_softmax(x.reshape(-1,self.k),dim=-1)
        x = x.reshape(batch_size,n_pts,self.k)
        return x


# In[25]:

# test whether the model work
if __name__ == '__main__':
    sim_data = torch.autograd.Variable(torch.rand(16,4,50))
    point_feat = MSPointNetFeat_V3(padding = 'same',
                                   dims_up = [64,128],
                                   dims_ms = [256],
                                   scales_conv = [3],
                                   scales_pool = []
                                  )
    out = point_feat(sim_data)
    print('point feat:',out.shape)
    
    seg = MSPointNetDenseCls_V3(padding = 'same',
                                dims_up = [64,128],
                                dims_ms = [256],
                                dims_down = [256,128],
                                scales_conv = [3],
                                scales_pool = []
                               )
    sim_data = sim_data.transpose(1,2)
    out = seg(sim_data)
    print('seg:',out.shape)


# In[ ]:




