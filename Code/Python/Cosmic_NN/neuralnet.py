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
from torch_baseloader import log
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

import matplotlib.pyplot as plt
import random
import math

class network(nn.Module):
    def __init__(self,batch_size = 10,num_classes = 2,   #Data for input/output of lin layer
                 in_channels = 1,out_channels_conv1 = 1,kernel_conv1 = 2,  #Conv layer 1
                 out_channels_conv2 = 1,kernel_conv2 = 2,   #Conv layer 2
                 pooling_kernel_1 = 16, #Pooling layer 1
                 pooling_kernel_2 = 4,  #Pooling layer 2
                 lin_input_size = 9,lin_output_size = 2): #lin layer
        '''
        define neural network
        '''
        super(network,self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.con1 = nn.Conv2d(in_channels,out_channels_conv1,kernel_conv1)
        self.pool1 = nn.MaxPool2d(pooling_kernel_1)
        self.con2 = nn.Conv2d(out_channels_conv1,out_channels_conv2,kernel_conv2)
        self.pool2 = nn.MaxPool2d(pooling_kernel_2)
        # ignore lin layer for now due to data sizes
        #before lin layer x.data.shape = [batchsize,9]
        self.lin = nn.Linear(lin_input_size,lin_output_size)
    def forward(self,x):
        #log("="*10+"Start of network"+"="*10)
        batsize = x.data.shape[0]
        #log(x.data.shape)
        x = self.con1(F.sigmoid(x))
        #log("="*10+"After conv1"+"="*10)
        #log(x.data.shape)
        x = self.pool1(F.relu(x))
        #log("="*10+"After pool1"+"="*10)
        #log(x.data.shape)
        x = self.con2(F.relu(x))
        #log("="*10+"After conv2"+"="*10)
        #log(x.data.shape)
        x = self.pool2(F.relu(x))
        #log("="*10+"After pool2"+"="*10)
        #log(x.data.shape)
        x = x.view(batsize,-1)
        #log("="*10+"After view"+"="*10)
        #log(x.data.shape)
        arg = F.softmax(F.relu(x),dim = 1)
        x = self.lin(arg)
        #log("="*10+"After lin"+"="*10)
        #log(x.data.shape)
        return x


if __name__ == "__main__":
    #detect which path to load data from
    cur_os = platform.system()
    log("Starting")
    log("OS detected: "+cur_os)
    datapath = ""
    settings = setup.load_settings()
    if cur_os == "Windows":
        datapath = settings["Spherharg"][4]
    elif cur_os == "Linux":
        datapath = settings["Spherharg"][5]
        datapath = datapath + "map_data\\"
    log("Using "+datapath+" as file path")
    #set up mailbot
    username = settings["Email"][4]
    password = settings["Email"][3]
    server = settings["Email"][2]

    bot = email_bot(settings["Email"][0],settings["Email"][1],server,password,username,int(settings["Email"][5]))

    #bot working
    #torch setup
    use_cuda = torch.cuda.is_available()
    log("Using CUDA: "+str(use_cuda))
    
    #####
    # Data Aqcuisition
    #####
    
    
    raw_data = bl.load_data(bl.load_filenames(datapath,"sub"))
    cutoff_percentage = 0.75    #How many percent to use for training/
    cutoff = int(cutoff_percentage*len(raw_data))
    raw_data_train = raw_data[:cutoff]
    raw_data_test = raw_data[cutoff:]
    norm_data_train = bl.normalize_data(raw_data_train,"0-1",False)
    norm_data_test = bl.normalize_data(raw_data_test,"0-1",False)
    log("Raw Data loaded. Turning to batches")
    
    batchsize = 10
    
    batches = bl.arr_to_batches(norm_data_train,batchsize,False)
    batches_test = bl.arr_to_batches(norm_data_test,batchsize,False)
    log("Batches generated")
    smaps_per_maps = 10#settings["NN"][0]
    log("Generating "+str(smaps_per_maps) +" string maps per stringless one.")
    
    G_mu = 10**-7
    v = 1
    A = G_mu*v
    train_arr = []
    
    percentage_with_strings = 0.5
    for batch in batches:
        stack = bl.create_map_array(batch,smaps_per_maps,G_mu,v,A,percentage_with_strings,False)
        train_arr.append(stack)
    log("Training Batches generated. "+str(len(train_arr)) +" Elements in train_arr.")
    test_arr = []
    for batch in batches_test:
        stack = bl.create_map_array(batch,smaps_per_maps,G_mu,v,A,percentage_with_strings,False)
        test_arr.append(stack)
    log("Testing Batches generated. "+str(len(test_arr)) + " Elements in test_arr.")
    #got all batches set correctly
    #this is now training and testing data
    
    
    ######
    #Network definition
    #####
    output_size = 2
    
    in_channels = 1
    out_channels_conv1 = 1
    kernel_conv1 = 2  #Conv layer 1
    out_channels_conv2 = 1
    kernel_conv2 = 2   #Conv layer 2
    pooling_kernel_1 = 16 #Pooling layer 1
    pooling_kernel_2 = 4  #Pooling layer 2
    lin_input_size = 9
    lin_output_size = 2
    net = network()
    print("="*10 + "NETWORK"+"="*10)
    print(net)
    print("="*27)
    
    if use_cuda:
        net = net.cuda()
    
    #####
    # Training parameters
    #####
    
    import torch.optim as optim
    crit = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),lr = 0.001,momentum = 0.9)
    log("Network and optimizers created")
    epochs = 1000
    train_losses = []
    t_train_start = datetime.now()
    for epoch in range(epochs):
        running_loss = 0
        net.train()
        #randomize training array
        random.shuffle(train_arr)
        for batch_id in range(len(train_arr)):  #cycle over all batches in train_arr
            batch = train_arr[batch_id]
            cur_maps = batch[0]
            idx = batch[1]
            temp_arr = []
            for m in cur_maps:
                temp_arr.append(m.data)
            in_map = Variable(torch.from_numpy(np.array(temp_arr)))
            classif = Variable(torch.from_numpy(idx))
            #log("In_map shape: "+str(in_map.data.shape))
            #log("Classifier shape: "+str(classif.data.shape))
            if use_cuda:
                in_map = in_map.cuda()
                classif = classif.cuda()
            
            in_map = in_map.unsqueeze(1)
            in_map = in_map.float()
            optimizer.zero_grad()
            pred = net(in_map)  
            #log("Output shape: "+str(pred.data.shape)) #matrix of [batchsize,numclasses] 
            
            loss = crit(pred.float(),classif.long())
            
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
            if round(batch_id/len(train_arr)*100) % 25 == 0:
                log("[Epoch: "+str(epoch)+"("+str(epoch/(epochs-1)*100)+"%): Data: "+str(batch_id/len(train_arr)*100)+"%]:Running loss: "+str(running_loss))
        train_losses.append(running_loss)
        log("="*20)
        log("Elapsed time since starting training: "+str(datetime.now() - t_train_start))    
        log("="*20)
    t_train_end = datetime.now()
    t_train_elapsed = t_train_end - t_train_start
    log("Elapsed time on training: "+str(t_train_elapsed))
        
    plt.plot(train_losses)
    plt.title("Train losses every batch vs datapoints: Epochs= "+str(epochs))
    #testing needed
    
    ######
    #Testing
    ######