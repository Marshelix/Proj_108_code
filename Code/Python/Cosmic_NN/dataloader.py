# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 01:32:31 2018

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

#Need a dataloader class for the CNN

class dataloader(Dataset):
    def __init__(self):
        super(dataloader, self).__init__()
    def __len__(self):
        return 0
    def __getitem__(self,idx):
        return 0

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
    print("CUDA available = "+str(use_cuda))
