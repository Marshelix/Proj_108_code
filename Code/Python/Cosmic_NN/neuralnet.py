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
log("="*40)
log("*"*40)
log("New Run")
log("*"*40)
log("="*40)

class network(nn.Module):
    #cifar
    def __init__(self,
                 in_channels = 1,out_channels_conv1 = 6,kernel_conv1 = 5,
                 out_channels_conv2 = 16,kernel_conv2 = 5,
                 pooling_kernel = 2,
                 lin_input_size = 43264,num_classes = 2):
        super(network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels_conv1, kernel_conv1)
        self.pool = nn.MaxPool2d(pooling_kernel, pooling_kernel)
        self.conv2 = nn.Conv2d(out_channels_conv1, out_channels_conv2, kernel_conv2)
        self.fc1 = nn.Linear(lin_input_size,num_classes)

    def forward(self, x):
        
        batsize = x.shape[0]
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(batsize, -1)
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = self.fc3(x)
       
        return x
    '''
    #def __init__(self,batch_size = 10,num_classes = 2,   #Data for input/output of lin layer
    #             in_channels = 1,out_channels_conv1 = 1,kernel_conv1 = 2,  #Conv layer 1
    #             out_channels_conv2 = 1,kernel_conv2 = 2,   #Conv layer 2
    #             pooling_kernel_1 = 16, #Pooling layer 1
    #             pooling_kernel_2 = 4,  #Pooling layer 2
    #             lin_input_size = 9,lin_output_size = 2): #lin layer
    '''
        #define neural network
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
    '''

if __name__ == "__main__":
    #detect which path to load data from
    will_save_model =False#for saving later
    print_freq = 50 #%. Output information during training when print_freq amounts are reached
    
    log("Saving system run = "+str(will_save_model))
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
    usage_percentage = 0.3
    cutoff_use = int(usage_percentage*len(raw_data))
    raw_data = raw_data[:cutoff_use]
    log("Using "+str(100*usage_percentage)+"% of data overall.")
    
    cutoff_percentage = 0.65    #How many percent to use for training/
    cutoff = int(cutoff_percentage*len(raw_data))
    raw_data_train = raw_data[:cutoff]
    raw_data_test = raw_data[cutoff:]
    log("Using "+str(100*cutoff_percentage)+"% of data for training.")
    
  
    
    log("Raw Data loaded. Turning to batches")
    
    batchsize = 10
    
    batches = bl.arr_to_batches(raw_data_train,batchsize,False)
    batches_test = bl.arr_to_batches(raw_data_test,batchsize,False)
    log("Batches generated")
    
    smaps_per_maps = 1#settings["NN"][0]
    log("Generating "+str(smaps_per_maps) +" string maps per stringless one/type of string.")
    
    
    ###
    # Projected time till completion
    ###
    time_per_map_gmu = 0.046399 #each map adds about 1.5 min

    Gmus = [1e-5,1e-6,1e-7,1e-8,1e-9,1e-10,1e-11]
    num_Gmus = len(Gmus)
    dt_gen = timedelta(seconds = time_per_map_gmu*smaps_per_maps*num_Gmus)
    log("Estimated time till completion of map generation: "+str(dt_gen))
    log("Estimated time of completion of map generation: "+str(datetime.now() + dt_gen))
    #Train network not just on one G_mu but multiple ones
    train_arr = []
    test_arr = []
    percentage_with_strings = 0.5
    log("Percent of maps with strings per batch: "+str(100*percentage_with_strings)+"%.")
    log("Amount of maps selected for strings per batch: "+str(percentage_with_strings*batchsize))
    t_g_start = datetime.now()
    for G_mu in Gmus:
        v = 1
        A = G_mu*v
        log("Values for string maps: (G_mu,v,A):("+str(G_mu)+","+str(v)+","+str(A)+")")
        for batch in batches:
            stack = bl.create_map_array(batch,smaps_per_maps,G_mu,v,A,percentage_with_strings,False,True)
            #stack = bl.normalize_data(np.array(stack),"0-1",True)
            train_arr.append(stack)
        log("Training Batches generated. "+str(len(train_arr)) +" Batches in train_arr.")
        
        log(str(len(train_arr[0][0]))+" elements per train batch.")
    
        for batch in batches_test:
            stack = bl.create_map_array(batch,smaps_per_maps,G_mu,v,A,percentage_with_strings,False,True)
            #stack = bl.normalize(stack,"0-1")
            test_arr.append(stack)
        log("Testing Batches generated. "+str(len(test_arr)) + " Batches in test_arr.")
        log(str(len(test_arr[0][0]))+" elements per test batch.")
    t_g_ela = datetime.now() - t_g_start
    tgela_per_map_gmu = t_g_ela/(num_Gmus*len(test_arr)*len(test_arr[0])*4)
    
 
    #train_arr[0][0][9].expand()
    log("Time elapsed on map generation: "+str(t_g_ela))
    log("Time elapsed per map in generation: "+str(tgela_per_map_gmu))
    #got all batches set correctly
    #this is now training and testing data
    log("#"*30)
    
    ######
    #Network definition
    #####
    output_size = 2
    
    img_size = 222
    
    
    in_channels = 1
    out_channels_conv1 = 1
    kernel_conv1 = 3  #Conv layer 1
    out_channels_conv2 = 1
    kernel_conv2 = 2   #Conv layer 2
    pooling_kernel_1 = 2 #Pooling layer 1
    pooling_kernel_2 = 2  #Pooling layer 2
    lin_input_size = in_channels*(int((((img_size - kernel_conv1 -1)/pooling_kernel_1)-kernel_conv2-1)/pooling_kernel_2)+1)**2
    lin_output_size = 2
    net =network()# network(batchsize,output_size,in_channels,out_channels_conv1,kernel_conv1,
#                  out_channels_conv2,
#                  kernel_conv2,
#                  pooling_kernel_1,pooling_kernel_2,lin_input_size,lin_output_size)
    print("="*10 + "NETWORK"+"="*10)
    print(net)
    print("="*27)
    log("#"*30)
    if use_cuda:
        net = net.cuda()
    
    #####
    # Training parameters
    #####
    
    import torch.optim as optim
    crit = nn.CrossEntropyLoss()
    lr = 1e-5
    mom = 0.9
    log("Optimizer definition:")
    log("le = "+str(lr)+", momentum = "+str(mom))
    optimizer = optim.SGD(net.parameters(),lr,momentum = mom)
    log("Network and optimizers created")
    log("#"*30)
    
    #######
    # Time till training completion
    #######
    
    epochs = 1000
    
    time_per_epoch_map = 0.003597  #s from test
    dt_train = timedelta(seconds = time_per_epoch_map * epochs*4*len(test_arr)*len(test_arr[0][0]))  #time estimate based on total time from experiment
    
    t_train_start = datetime.now()
    t_train_finish_proj = t_train_start + dt_train
    log("Projected finishing time = "+str(t_train_finish_proj))
    log("Projected time to completion = "+str(dt_train))
    
    train_losses = []
    test_losses = []
    correctness = []
    # correctness_per_class: array of arrays for all classes
    correctness_per_class = []
    for i in range(num_Gmus):
        correctness_per_class.append([])
    
    pred_changed = []
    fig,(ax1,ax2,ax3,ax4,ax5) = plt.subplots(5,1)
    fig.tight_layout()
    ###########################################################################
    #Setup end
    ###########################################################################
    def train(epoch):
        '''
        Train the neural net, update plots accordingly
        '''
        running_loss = 0
        net.train()
        #randomize training array
        random.shuffle(train_arr)
        for batch_id in range(len(train_arr)):  #cycle over all batches in train_arr
            '''
            batch = train_arr[batch_id]
            cur_maps = batch[0]
            idx = batch[1]
            '''


            cur_maps,idx = train_arr[batch_id]
            
            temp_arr = []
            for m in cur_maps:
                temp_arr.append(m.data)
            in_map,classif = Variable(torch.from_numpy(np.array(temp_arr))),Variable(torch.from_numpy(idx))
            #log("In_map shape: "+str(in_map.data.shape))
            #log("Classifier shape: "+str(classif.data.shape))
            if use_cuda:
                in_map = in_map.cuda()
                classif = classif.cuda()
            
            in_map = in_map.unsqueeze(1)
            in_map = in_map.float()
            
            pred = net(in_map)  
            #log("Output shape: "+str(pred.data.shape)) #matrix of [batchsize,numclasses] 
            loss = crit(pred.float(),classif.long())
            
            optimizer.zero_grad()
            loss.backward()
            #print("Loss: ")
            #print(loss)
        
            #print("Parameters Gradients: ")
            #for param in net.parameters():
            #    print(param.grad.data.sum())

            # start debugger
            #import pdb; pdb.set_trace()

            optimizer.step()

           
            # Output changed?
            #print("Prediction after step changed?: ")
            #print((net(in_map) == pred).sum() != 0)
            pred_changed.append(((net(in_map) == pred).sum() != 0).data)
            running_loss += loss.item()
            #for param in net.parameters():
            #    print(param.grad.data.sum())
            if int(batch_id/len(train_arr)*100) % print_freq == 0:
                log("[Epoch: "+str(epoch)+"("+str(epoch/max((epochs-1),1)*100)+"%): Data: "+str(batch_id/len(train_arr)*100)+"%]:Running loss: "+str(running_loss))
                #log("[Epoch: "+str(epoch)+"("+str(epoch/max((epochs-1),1)*100)+"%): Data: "+str(batch_id/len(train_arr)*100)+"%]:String maps percentage: "+str(100*classif.sum().data/len(classif))+"%")
                # != accuracy
        train_losses.append(running_loss)
        ax1.clear()
        ax1.plot(train_losses)
        
        ax1.set_title("Train losses every batch vs datapoints: Epoch #"+str(epoch)+" => " +str(running_loss))
        ax4.clear()
        ax4.plot(train_losses/np.max(train_losses))
        ax4.set_title("Relative change in losses")
    def test(epoch):
        #Run testing for monitoring
    
        net.eval()
        test_loss_train = 0
        correct = 0
        total_test = 0  #total num tested maps
        num_tested = 0
        for batch_id in range(len(test_arr)):
            batch = test_arr[batch_id]
            cur_maps = batch[0]
            idx = batch[1]
            temp_arr = []
            for m in cur_maps:
                temp_arr.append(m.data)
            in_map = Variable(torch.from_numpy(np.array(temp_arr)))
            #print("Classifier: ")
            
            classif = Variable(torch.from_numpy(idx))
            #print(classif)
            if use_cuda:
                in_map = in_map.cuda()
                classif = classif.cuda()
            in_map = in_map.unsqueeze(1)
            in_map = in_map.float()
        
            pred = net(in_map)
            loss = crit(pred.float(),classif.long())
            test_loss_train += loss.item()
            classif = classif.long()
            pred_class = pred.data.max(1,keepdim = True)[1] #max index
            pred_class = pred_class.long()
            total_test += batchsize
            correct += pred_class.eq(classif.data.view_as(pred_class)).long().cpu().sum()
            #print("Correctness Values: ")
            num_tested += len(cur_maps)
            #print(pred_class.eq(classif.data.view_as(pred_class)).long())
        log("Test set accuracy: "+str(100*correct.item()/total_test) + "% ,loss = "+str(test_loss_train))
        log("Correct hits: "+str(correct.item())+"/"+str(total_test))
        correctness.append(100*correct/total_test)    
        test_losses.append(test_loss_train)
        ax2.clear()
        ax2.plot(test_losses)
        ax2.set_title("Test losses every batch: Epoch #"+str(epoch)+" => " +str(test_loss_train))
    
        ax3.clear()
        ax3.plot(correctness)
        ax3.set_title("Accuracy @ Epoch #: "+str(epoch)+" => " +str(100*correct.item()/num_tested)+"%")      
        
        ax4.plot(test_losses/np.max(test_losses))
        plt.pause(1e-7)
        
        calc_accuracy_per_class(epoch)  #calculate accuracies for each class and plot
        
        fig.canvas.set_window_title(str(100*epoch/epochs)+"%")
        log("="*20)
        log("Elapsed time since starting training: "+str(datetime.now() - t_train_start))
        log("Estimated time left: "+str(t_train_finish_proj - datetime.now()))
        log("="*20)
    
    def calc_accuracy_per_class(epoch):
        '''
        Calculates the accuracies for each class like the testing algo
        Plots to ax5
        '''
        batches_per_class = len(test_arr)/num_Gmus
        net.eval()#just in case
        for i in range(num_Gmus):
            '''
            select the ith G_mu
            '''
            accura = 0
            test_loss_class = 0
            lower = i*batches_per_class
            upper = (i+1)*batches_per_class
            #testing is not shuffled, 
            #ie we expect the test arrays to consist of batches of only one class each
            # -> can loop through classes
            num_tested = 0
            for batch_id in range(int(lower),int(upper)):
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
                test_loss_class += loss.item()
                classif = classif.long()
                pred_class = pred.data.max(1,keepdim = True)[1] #max index
                pred_class = pred_class.long()
                accura += pred_class.eq(classif.data.view_as(pred_class)).long().cpu().sum()
                num_tested += len(cur_maps)
            log("Accuracy and Loss for "+str(Gmus[i])+": "+str(100*accura.item()/(num_tested))+"%"+", loss = "+str(test_loss_class))
            log("Hits: "+str(accura.item())+"/"+str(num_tested))
            correctness_per_class[i].append(100*accura.item()/(num_tested))
        ax5.clear()
        for elem in correctness_per_class:    
            ax5.plot(elem)
        ax5.set_title("Accuracies per class")
        ax5.legend([str(x) for x in Gmus],loc = 'upper center',bbox_to_anchor = (0.5,1.01),ncol = num_Gmus)  #legend next to plot
        plt.pause(1e-7)
    
    
    for epoch in range(epochs):
        train(epoch)
        test(epoch)
    t_train_end = datetime.now()
    t_train_elapsed = t_train_end - t_train_start
    
    log("Elapsed time on training: "+str(t_train_elapsed))
    log("Elapsed time per Epoch/map: "+str(t_train_elapsed/(epochs*4*len(test_arr)*len(test_arr[0][0]))))
    
    
    #=========================================================================#
    #=========================================================================#
    ######
    #Testing
    ######
    test(epochs)
    '''
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
        
        loss = F.cross_entropy(pred.float(),classif.long())
        test_loss += loss.data[0]
        classif = classif.long()
        pred_class = pred.data.max(1,keepdim = True)[1] #max index
        pred_class = pred_class.long()
        correct += pred_class.eq(classif.data.view_as(pred_class)).long().cpu().sum()
    log("Test set accuracy: "+str(100*correct/(len(test_arr)*len(test_arr[0][0]))) + "% ,loss = "+str(test_loss))
    '''
        
    ######
    # Saving
    ######
    if will_save_model:
        f_savename = "Models/Model_"+str(epochs)+"_"+str(num_Gmus)+"_"+str(int(100*usage_percentage))
        with open(f_savename+"_model.dat","wb") as f:
            torch.save(net,f)
        #Save state dicts
        with open(f_savename+"_net_dict.dat","wb") as f:
            torch.save(net.state_dict(),f)
        with open(f_savename+"_opti_dict.dat","wb") as f:
            torch.save(optimizer.state_dict(),f)
        fig.savefig(f_savename + "_figures.png")
    
    
    ############
    #Accuracy for different classes
    #Already tested in test above by now
    ############
    '''
    batches_per_class = len(test_arr)/num_Gmus
    net.eval()

    
    accuracies_per_class = []
    test_losses = []
    for i in range(num_Gmus):
        accura = 0
        test_loss_class = 0
        lower = i*batches_per_class
        upper = (i+1)*batches_per_class
        #testing is not shuffled, 
        #ie we expect the test arrays to consist of batches of only one class each
        # -> can loop through classes
        num_tested = 0
        for batch_id in range(int(lower),int(upper)):
            batch = test_arr[batch_id]
            cur_maps = batch[0]
            idx = batch[1]
            temp_arr = []
            for m in cur_maps:
                temp_arr.append(m.data)
            in_map = Variable(torch.from_numpy(np.array(temp_arr)))
            classif = Variable(torch.from_numpy(idx))#
            if use_cuda:
                in_map = in_map.cuda()
                classif = classif.cuda()
            in_map = in_map.unsqueeze(1)
            in_map = in_map.float()
        
            pred = net(in_map)
        
            loss = crit(pred.float(),classif.long())
            test_loss_class += loss.item()
            classif = classif.long()
            pred_class = pred.data.max(1,keepdim = True)[1] #max index
            pred_class = pred_class.long()
            accura += pred_class.eq(classif.data.view_as(pred_class)).long().cpu().sum()
            num_tested += len(cur_maps)
        log("Accuracy and Loss for "+str(Gmus[i])+": "+str(100*accura.item()/(num_tested))+"%"+", loss = "+str(test_loss_class))
        log("Hits: "+str(accura.item())+"/"+str(num_tested))
        test_losses.append(test_loss_class)
        accuracies_per_class.append(100*accura.item()/(num_tested))
    '''
    log("*"*30)
    log("Run data: ")
    log("Epochs = "+str(epochs))
    log("Optimizer: ")
    log("lr = "+str(lr))
    log("Momentum = "+str(mom))
    log("Program settings: ")
    log("Saving model = "+str(will_save_model))
    log("Batchsize = "+str(batchsize))
    log("Using "+str(100*usage_percentage)+"% of input data")
    log("String percentage = "+str(100*percentage_with_strings)+"%")
    log("Amount of string classes = "+str(num_Gmus))
    log("Amount of string implanted maps from base maps= "+str(smaps_per_maps))
    log("Split of training data/test data: "+str(100*cutoff_percentage)+"%/"+str(100*(1-cutoff_percentage))+"%")
    log("*"*30)
            
    #TODO:
    #Write loader for network