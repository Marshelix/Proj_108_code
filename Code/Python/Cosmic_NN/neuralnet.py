# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 01:38:30 2018

@author: martin
"""

import torch
from torch.autograd import Variable
import torchvision
from torch.utils.data import Dataset
import platform
import os
import sys
import numpy as np
sys.path.append(os.path.dirname(__file__))
import torch_baseloader as bl
mailpath = os.path.abspath("../py_mail")
sys.path.append(mailpath)
sys.path.append(os.path.abspath("../Setup"))
from mailbot import email_bot

from Setup import setup
import pickle
from datetime import datetime
import pyshtools as sht
from dataloader import dataloader
from torch import nn
import torch.nn.functional as F


classes = [0,1]


class network(nn.Module):
    def __init__(self,batch_size, input_size,h1_size,h2_size,h3_size,output_size):
        '''
        define neural network
        '''
        super(network,self).__init__()
        self.in_size = input_size
        self.out_size = output_size
        self.con1 = nn.Conv2d(batch_size,h1_size,input_size)
        self.pool1 = nn.MaxPool2d(2,2)
        self.con2 = nn.Conv2d(batch_size,h2_size,h1_size)
        self.pool2 = nn.MaxPool2d(2,2)
        self.l1 = nn.Linear(h2_size,h3_size)
        self.l2 = nn.Linear(h3_size,output_size)
        
    def forward(self,x):
        x = self.con1(F.sigmoid(x))
        x = self.pool1(F.relu(x))
        x = self.con2(F.relu(x))
        x = self.pool2(F.relu(x))
        x = self.l1(F.relu(x))
        x = self.l2(F.relu(x))
        return x


if __name__ == "__main__":
    #detect which path to load data from
    cur_os = platform.system()
    print("OS detected: "+cur_os)
    datapath = ""
    settings = setup.load_settings()
    if cur_os == "Windows":
        datapath = settings["Spherharg"][4]
    elif cur_os == "Linux":
        datapath = settings["Spherharg"][5]
        datapath = datapath + "map_data\\"
    print("Using "+datapath+" as file path")
    #set up mailbot
    username = settings["Email"][4]
    password = settings["Email"][3]
    server = settings["Email"][2]

    bot = email_bot(settings["Email"][0],settings["Email"][1],server,password,username,int(settings["Email"][5]))

    #bot working
    #torch setup
    use_cuda = torch.cuda.is_available()
    print("Using CUDA: "+str(use_cuda))
    
    raw_data = bl.load_data(bl.load_filenames(datapath,"sub"))
    raw_data = raw_data[:int(0.1*len(raw_data))]
    print(len(raw_data))
    norm_data = bl.normalize_data(raw_data,"0-1",False)
    print(len(norm_data))
    print("Raw Data loaded. Turning to batches")
    batches = bl.arr_to_batches(norm_data,10,True)
    
    smaps_per_maps = 1#settings["NN"][0]
    G_mu = 10**-7
    v = 1
    A = G_mu*v
    arr = []
    for batch in batches:
        stack = bl.create_map_array(batch,smaps_per_maps,G_mu,v,A,0.5,True)
        arr.append(stack)
        print(len(stack))
    print(arr)