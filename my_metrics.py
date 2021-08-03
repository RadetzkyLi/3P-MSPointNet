#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix




def metrics(y_pred,y_true,method = 'accuracy',average=None):
    '''
    Caculate some criterion according to method, including 
    accuracy,precision,recall,f1_score.
    :param method : criterion method;
    :param average : required for multi-class targets,default is None,
        if None,scores of each class are returned;
        if micro,caculate metrics globally;
        if macro,caculate unweighted mean score of each label; 
    '''
    if method == 'accuracy':
        return accuracy_score(y_true,y_pred)
    elif method == 'precision':
        return precision_score(y_true,y_pred,average=average,labels=[0,1,2,3,4])
    elif method == 'recall':
        return recall_score(y_true,y_pred,average=average,labels=[0,1,2,3,4])
    elif method == 'f1_score':
        return f1_score(y_true,y_pred,average=average,labels=[0,1,2,3,4])
    elif method == 'confusion_matrix':
        return confusion_matrix(y_true,y_pred)
    else:
        raise ValueError('Unsupported evaluate method:',method)


# In[ ]:




