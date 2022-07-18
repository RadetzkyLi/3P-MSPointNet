#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn


# In[5]:


if __name__ == '__main__':
    try:
        # this is shell comman!
        get_ipython().system('jupyter nbconvert --to python new_layers.ipynb   ')
    except:
        pass


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
        elif self.padding == 'causal':
            padding = 0
            stride = 1
            dilation = 1
        else:
            raise ValueError('Unexpected padding method:',self.padding)
        maxpool = nn.MaxPool1d(kernel_size,
                               stride = stride,
                               dilation = dilation,
                               padding = padding
                              )
        return maxpool
        
    def casual_maxpool1d(self,x):
        padding_dim = self.kernel_size - 1
        if len(x.shape) == 3:
            padding_tensor = torch.zeros((x.shape[0],x.shape[1],padding_dim))
            padding_tensor = padding_tensor.to(x.device)
            x_new = torch.cat((padding_tensor,x),dim=2)
#             print(x_new)
#         elif len(x.shape) == 2:
#             padding_tensor = torch.zeros((x.shape[0],padding_dim))
#             padding_tensor = padding_tensor.to(x.device)
#             x_new = torch.cat((padding_tensor,x),dim=1)
        else:
            raise ValueError('input shape can only be 2-axis or 3-axis,got ',x.shape)
        return self.maxpool(x_new)
        
    def forward(self,x):
        # in:[B,C,L]
        if self.padding == 'casual':
            x = self.casual_maxpool1d(x)
        else:
            x = self.maxpool(x)
        return x


# In[27]:


if __name__ == '__main__':
    conv = NewMaxPool1d(3,stride=1,padding='same',dilation=1)
    sim_data = torch.autograd.Variable(torch.rand(16,4,250))
    out = conv(sim_data)
    print(out.shape)

