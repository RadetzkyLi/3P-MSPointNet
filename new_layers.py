#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn


# # 1.NewConv1d

# In[20]:


class NewConv1d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,
                padding = 'valid',dilation=1,padding_mode='zeros'):
        super(NewConv1d,self).__init__()
        self.padding = padding
        self.padding_mode = padding_mode
        self.conv1d = self.get_conv1d(in_channels,out_channels,kernel_size,
                                      stride=stride,dilation=dilation)
    
    def get_conv1d(self,in_channels,out_channels,kernel_size,stride=1,dilation=1):
        if self.padding == 'valid':
            padding = 0
        elif self.padding == 'same':
            padding = kernel_size//2
        elif self.padding == 'casual':
            padding = kernel_size - 1
        else:
            raise ValueError('Unexpected padding method:',self.padding)
        conv1d = nn.Conv1d(in_channels,out_channels,kernel_size,
                             stride = stride,
                             dilation = dilation,
                             padding_mode = self.padding_mode,
                             padding = padding)
        return conv1d
        
    def forward(self,x):
        # in:[B,C,L]
        dim = x.shape[-1]
        x = self.conv1d(x)
        if self.padding == 'casual':
            x = x[:,:,:dim]
        return x


# In[21]:


if __name__ == '__main__':
    conv = NewConv1d(4,32,3,padding='casual',stride=2,dilation=1)
    sim_data = torch.autograd.Variable(torch.rand(16,4,250))
    out = conv(sim_data)
    print(out.shape)


# # 2.New MaxPool1d

# In[22]:


class NewMaxPool1d(nn.Module):
    def __init__(self,kernel_size,stride=1,
                padding = 'valid',dilation=1):
        super(NewMaxPool1d,self).__init__()
        self.padding = padding
        self.maxpool = self.get_maxpool1d(kernel_size,
                                          stride=stride,dilation=dilation)
    
    def get_maxpool1d(self,kernel_size,stride=1,dilation=1):
        if self.padding == 'valid':
            padding = 0
        elif self.padding == 'same':
            padding = kernel_size//2
        elif self.padding == 'casual':
            padding = kernel_size//2
        else:
            raise ValueError('Unexpected padding method:',self.padding)
        maxpool = nn.MaxPool1d(kernel_size,
                               stride = stride,
                               dilation = dilation,
                               padding = padding
                              )
        return maxpool
        
    def forward(self,x):
        # in:[B,C,L]
        dim = x.shape[-1]
        x = self.maxpool(x)
        return x


# In[27]:


if __name__ == '__main__':
    conv = NewMaxPool1d(3,stride=1,padding='same',dilation=1)
    sim_data = torch.autograd.Variable(torch.rand(16,4,250))
    out = conv(sim_data)
    print(out.shape)


# In[ ]:




