# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 01:38:30 2018

@author: marti
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
mailpath = os.path.abspath(os.path.join(os.path.dirname(__file__),".."+"/py_mail"))
sys.path.append(mailpath)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),".."+"/Setup")))
from mailbot import email_bot
from Setup import setup
import pickle
from datetime import datetime
import pyshtools as sht
from dataloader import dataloader
from torch import nn
import torch.nn.functional as F

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
        self.pool2 = nn.maxPool2d(2,2)
        self.l1 = nn.linear(h2_size,h3_size)
        self.l2 = nn.linear(h3_size,output_size)
        
    def forward(self,x):
        x = self.con1(x)
        x = self.pool1(x)
        x = self.con2(F.relu(x))
        x = self.pool2(x)
        x = self.l1(x)
        x = self.l2(x)
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
