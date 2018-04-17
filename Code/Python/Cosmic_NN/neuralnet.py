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
from datetime import datetime,timedelta

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
    
    smaps_per_maps = 25#settings["NN"][0]
    log("Generating "+str(smaps_per_maps) +" string maps per stringless one.")
    
    
    ###
    # Projected time till completion
    ###
    time_per_map = 100   #each map adds about 1.5 min

    
    dt_gen = timedelta(seconds = time_per_map*smaps_per_maps)
    log("Estimated time till completion of map generation: "+str(dt_gen))
    log("Estimated time of completion of map generation: "+str(datetime.now() + dt_gen))
    
    
    
    G_mu = 10**-7
    v = 1
    A = G_mu*v
    log("Values for string maps: (G_mu,v,A):("+str(G_mu)+","+str(v)+","+str(A)+")")
    train_arr = []
    
    percentage_with_strings = 0.5
    for batch in batches:
        stack = bl.create_map_array(batch,smaps_per_maps,G_mu,v,A,percentage_with_strings,False)
        train_arr.append(stack)
    log("Training Batches generated. "+str(len(train_arr)) +" Elements in train_arr.")
    log(str(len(train_arr[0][0]))+" elements per train batch.")
    test_arr = []
    for batch in batches_test:
        stack = bl.create_map_array(batch,smaps_per_maps,G_mu,v,A,percentage_with_strings,False)
        test_arr.append(stack)
    log("Testing Batches generated. "+str(len(test_arr)) + " Elements in test_arr.")
    log(str(len(test_arr[0][0]))+" elements per test batch.")
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
    optimizer = optim.SGD(net.parameters(),lr = 0.001,momentum = 0)
    log("Network and optimizers created")
    
    #######
    # Time till training completion
    #######
    
    epochs = 1000
    
    time_per_epoch = 7.1  #s
    dt_train = timedelta(seconds = time_per_epoch * epochs)
    
    t_train_start = datetime.now()
    t_train_finish_proj = t_train_start + dt_train
    log("Projected finishing time = "+str(t_train_finish_proj))
    log("Projected time to completion = "+str(dt_train))
    
    train_losses = []
    test_losses = []
    correctness = []
    f,(ax1,ax2,ax3) = plt.subplots(3,1)
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
            if math.floor(batch_id/len(train_arr)*100) % 25 == 0:
                log("[Epoch: "+str(epoch)+"("+str(epoch/max((epochs-1),1)*100)+"%): Data: "+str(batch_id/len(train_arr)*100)+"%]:Running loss: "+str(running_loss))
        train_losses.append(running_loss)
        ax1.clear()
        ax1.plot(train_losses)
        
        ax1.set_title("Train losses every batch vs datapoints: Epoch #"+str(epoch))
        
        ####
        #
        # Perform testing here!
        #
        ####
        net.eval()
        test_loss_train = 0
        correct = 0
        for batch_id in range(len(test_arr)):
            batch = test_arr[batch_id]
            cur_maps = batch[0]
            idx = batch[1]
            temp_arr = []
            for m in cur_maps:
                temp_arr.append(m.data)
            in_map = Variable(torch.from_numpy(np.array(temp_arr)))
            classif = Variable(torch.from_numpy(idx))
            if use_cuda:
                in_map = in_map.cuda()
                classif = classif.cuda()
            in_map = in_map.unsqueeze(1)
            in_map = in_map.float()
        
            pred = net(in_map)
            loss = crit(pred.float(),classif.long())
            test_loss_train += loss.data[0]
            classif = classif.long()
            pred_class = pred.data.max(1,keepdim = True)[1] #max index
            pred_class = pred_class.long()
            correct += pred_class.eq(classif.data.view_as(pred_class)).long().cpu().sum()
        log("Test set accuracy: "+str(100*correct/(len(test_arr)*len(test_arr[0][0]))) + "% ,loss = "+str(test_loss_train))
        correctness.append(100*correct/(len(test_arr)*len(test_arr[0][0])))    
        test_losses.append(test_loss_train)
        ax2.clear()
        ax2.plot(test_losses)
        ax2.set_title("Test losses every batch: Epoch #"+str(epoch))
        
        ax3.clear()
        ax3.plot(correctness)
        ax3.set_title("Accuracy @ Epoch #: "+str(epoch))              
        plt.pause(1e-7)
        
        log("="*20)
        log("Elapsed time since starting training: "+str(datetime.now() - t_train_start))
        log("Estimated time left: "+str(t_train_finish_proj - datetime.now()))
        log("="*20)
    t_train_end = datetime.now()
    t_train_elapsed = t_train_end - t_train_start
    log("Elapsed time on training: "+str(t_train_elapsed))

    #testing needed
    
    ax1.clear()
    ax1.plot(train_losses)
    ax1.set_title("Train losses every batch vs datapoints: Epoch #"+str(epoch))
    
    ax2.clear()
    ax2.plot(test_losses)
    ax2.set_title("Test losses every batch: Epoch #"+str(epoch))
        
    ax3.clear()
    ax3.plot(correctness)
    ax3.set_title("Accuracy @ Epoch #: "+str(epoch))              
    ######
    #Testing
    ######
    
    
    net.eval()
    test_loss = 0
    correct = 0
    for batch_id in range(len(test_arr)):
        batch = test_arr[batch_id]
        cur_maps = batch[0]
        idx = batch[1]
        temp_arr = []
        for m in cur_maps:
            temp_arr.append(m.data)
        in_map = Variable(torch.from_numpy(np.array(temp_arr)))
        classif = Variable(torch.from_numpy(idx))
        if use_cuda:
            in_map = in_map.cuda()
            classif = classif.cuda()
        in_map = in_map.unsqueeze(1)
        in_map = in_map.float()
        
        pred = net(in_map)
        
        loss = crit(pred.float(),classif.long())
        test_loss += loss.data[0]
        classif = classif.long()
        pred_class = pred.data.max(1,keepdim = True)[1] #max index
        pred_class = pred_class.long()
        correct += pred_class.eq(classif.data.view_as(pred_class)).long().cpu().sum()
    log("Test set accuracy: "+str(100*correct/(len(test_arr)*len(test_arr[0][0]))) + "% ,loss = "+str(test_loss))
    # Saving
    ######
    with open("Model_"+str(epochs)+".dat","wb") as f:
        torch.save(net,f)