# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 18:25:42 2018

@author: Martin Sanner
"""

import torch
from torch.autograd import Variable
import torchvision
import platform
import os
import sys
import numpy as np
mailpath = os.path.abspath(os.path.join(os.path.dirname(__file__),".."+"/py_mail"))
sys.path.append(mailpath)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),".."+"/Setup")))
from mailbot import email_bot
from Setup import setup
import pickle
from datetime import datetime
import pyshtools as sht

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


def load_filenames(path, specifier = None,b_Verbose = False):
    '''
    returns all filenames in path
    '''
    if specifier is not None:
        arr = []
        for file in os.listdir(path):
            if specifier in file:
                arr.append(path + file)
        if b_Verbose:
            print(arr)
        return arr
    else:
        if b_Verbose:
            print(os.listdir(path))
        return path + os.listdir(path)

def load_data(filenames,b_Verbose = False):
    '''
    return: all files in datapath
    '''
    total_arr = []
    for name in filenames:
        f = open(name,"rb")
        if b_Verbose:
            print(name + " opened.")
        map_arr = pickle.load(f)
        for m in map_arr:
            #array of
            total_arr.append(m)
        f.close()
        if b_Verbose:
            print(name + " closed.")
    
    return np.array(total_arr)
def normalize_data(data,normalization = "1-1",b_Verbose = False):
    '''
    Normalizes a set of data based on a type of normalization, either:
        data = np.array of data to normalize
        "1-1":
            to a range of [-1,1]
        "0-1":
            to a range of [0,1]
    '''
    
    if b_Verbose:
        print("normalization = "+normalization)
        print(normalization == "1-1")
        print(normalization == "0-1")
    new_arr = []
    if normalization == "1-1" or normalization == "0-1":
        if normalization == "1-1":
            for elem in data:
                minmax = np.max(elem.data) - np.min(elem.data)
                elem1 = (elem.data-np.min(elem.data))/minmax
                elem2 = 2*elem1 -1 
                new_arr.append(sht.SHGrid.from_array(elem2))
                
                if b_Verbose:
                    print("min: "+str(np.min(elem2.data)))
                    print("max: "+str(np.max(elem2.data)))
        elif normalization == "0-1":
            for elem in data:
                minmax = np.max(elem.data)-np.min(elem.data)
                elem1 = (elem.data-np.min(elem.data))/minmax
                if b_Verbose:
                    print("min: "+str(np.min(elem1.data)))
                    print("max: "+str(np.max(elem1.data)))
                new_arr.append(sht.SHGrid.from_array(elem1))
    
    else:
        print("Unknown normalization. Returning original set")
        return data
    return np.array(new_arr)
    

def dataarr_to_tensor_stack(data,b_Verbose = False):
    '''
    Converts an array of data into a pytorch stack
    '''
    t_start = datetime.now()
    #transfer to torch array
    t_arr = []
    i = 1
    for elem in norm_dat:
        e1 = Variable(torch.from_numpy(elem.data)).cuda() if use_cuda else Variable(torch.from_numpy(elem.data)) 
        t_arr.append(e1)
        if b_Verbose:        
            print("Elem appended - "+str(100*(i/len(norm_dat)))+"% completed - iteration #"+str(i))
            i = i+1
    if b_Verbose:
        print("Converting to np array")
    t_arr = torch.stack(t_arr).cuda()
    if b_Verbose:
        print("Elements appended")
    t_elapsed = datetime.now() - t_start
    
    print("Elapsed time on conversion: "+str(t_elapsed))
    return t_arr
    
if __name__ == "__main__":
    t_start = datetime.now()
    data = load_data(load_filenames(datapath,"sub"))
    print(len(data))
    
    norm_dat = normalize_data(data)
    
    t_elapsed = datetime.now() - t_start
    
    print("Elapsed time one file loading: "+str(t_elapsed))
    dataarr_to_tensor_stack(norm_dat,False)