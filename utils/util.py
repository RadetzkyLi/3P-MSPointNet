#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
from datetime import datetime
import time
import csv
import numpy as np
import pickle



def save_list2txt(file_name,list_data):
    file = open(file_name,'w')
    file.write(str(list_data))
    file.close()
    
def save_list2csv(file_name,columns,list_data):
    '''
    Save list data as .csv file.
    :param file_name: csv file name such as 'test.csv';
    :param columns: list data,headers of csv file such as '['lat','lon']';
    :param list_data: list data ,such as [[1,2,...],...];
    :return : None.
    '''
    with open(file_name,'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(list_data)
        f.close()

def load_csv2list(file_name,is_traj=True):
    '''
    Load csv file as list.
    :param file_name:csv file name such as 'test.csv',
        the csv file must have headers as first row.
    :param is_traj:whether the file is trajectory,default is True;
    '''
    def process_row(row_data):
        if not is_traj:
            data = [float(row_data[i]) for i in range(n_columns)]
        elif is_traj and n_columns==4:
            data = [float(row[0]),float(row[1]),float(row[2]),int(row[3])]
        else:
            raise ValueError('Value of is_traj is not consistent with n_columns.')
        return data
    list_data = []
    with open(file_name) as f:
        csv_reader = csv.reader(f)
        columns = next(csv_reader)
        n_columns = len(columns)
        cnt = 1
        for row in csv_reader:
            try:
                list_data.append(process_row(row))
                cnt += 1
            except:
                raise ValueError('Fail to read rows:',cnt,' in file:',file_name)
        f.close()
    return list_data


# In[ ]:


def load_pickle(file_path):
    '''
    Load pickle data of file path.
    '''
    with open(file_path,'rb') as f:
        data = pickle.load(f)
    return data


# In[4]:


def days_date(time_str):
    date_format = "%Y/%m/%d %H:%M:%S"
    current = datetime.strptime(time_str,date_format)
    date_format = "%Y/%m/%d"
    bench = datetime.strptime('1899/12/30',date_format)
    no_days = current - bench
    delta_time_days = no_days.days + current.hour / 24.0 +        current.minute / (24.*60.) + current.second / (24.*3600.)
    return delta_time_days


# In[2]:


def timestamp2str(timestamp,time_format='%Y-%m-%d %H:%M:%S'):
    delta = days_date('1970/01/01 00:00:00')
    now_time = datetime.utcfromtimestamp((timestamp-delta)*24*3600)
    return now_time.strftime(time_format)


# In[ ]:


Mode2Index = {"walk":0,"run":8,"bike":1,"bus":2,"car":3,"taxi":4,
              "subway":5,"railway":6,"train":7,"motocycle":8,"boat":8,
             "airline":8,"other":8}
Index2Color = {0:'red',1:'beige',2:'blue',3:'green',4:'orange',5:'pink',6:'purple',7:'gray',8:'white'}


# In[ ]:




