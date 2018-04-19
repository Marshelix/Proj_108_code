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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),".."+"/spherical_harmonics_generator")))
from spher_harmonics_funcs import add_strings
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
    datapath = datapath #+ "map_data\\"
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
        return [path + x for x in os.listdir(path)]

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
def normalize_data(data,normalization = "1-0",b_Verbose = False):
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
    use_cuda = torch.cuda.is_available()    #just in case
    #transfer to torch array
    t_arr = []
    i = 1
    for elem in data:
        e1 = Variable(torch.from_numpy(elem.data)).cuda() if use_cuda else Variable(torch.from_numpy(elem.data)) 
        t_arr.append(e1)
        if b_Verbose:        
            print("Elem appended - "+str(100*(i/len(data)))+"% completed - iteration #"+str(i))
            i = i+1
    if b_Verbose:
        print("Converting to np array")
    t_arr = torch.stack(t_arr).cuda() if use_cuda else torch.stack(t_arr)
    if b_Verbose:
        print("Elements appended")
    t_elapsed = datetime.now() - t_start
    
    print("Elapsed time on conversion: "+str(t_elapsed))
    return t_arr

def create_string_maps(base_maps,num_smaps_per_map,G_mu,V,Amp,b_Verbose = False):
    '''
    Creates an array of map representations with strings implanted. 
    Since these are all saved in memory, it is importent to recognize how many representations we can do safely.
    For 10gb assigned, we have ~25 reps. For 25gb available, we have ~60 reps.
    
    '''
    arr = []
    if b_Verbose:
        print("#maps = "+str(num_smaps_per_map))
        print("G_mu = "+str(G_mu))
        print("v = "+str(V))
        print("Amplitude = "+str(Amp))
    for cur_map in base_maps:
        for i in range(num_smaps_per_map):
            _,smap = add_strings(cur_map,G_mu,V,1,None,None,Amp) #string map that is not saved to file
            arr.append(smap)
    arr = np.array(arr)
    if b_Verbose:
        print("Amount of maps in array: "+str(len(arr)))
    return dataarr_to_tensor_stack(arr,b_Verbose)

def create_map_stack(base_maps,num_smaps_per_map,G_mu,V,Amp,percentage_string = 0.5,b_Verbose = False):
    '''
    Creates 2 stacks, one with strings and one without. Takes the 
    '''
    np.random.shuffle(base_maps)    #array now shuffled
    max_i = int(percentage_string* len(base_maps))-1
    if b_Verbose:
        print("Max_i = "+str(max_i))
    string_maps = base_maps[0:max_i]
    staying_maps = base_maps[max_i:]
    string_maps = create_string_maps(string_maps,num_smaps_per_map,G_mu,V,Amp,b_Verbose)
    staying_maps = dataarr_to_tensor_stack(staying_maps,b_Verbose)
    #join stacks
    return torch.cat((staying_maps,string_maps))
    
def create_string_maps_arr(base_maps,num_smaps_per_map,G_mu,V,Amp,b_Verbose = False):
    '''
    Creates an array of map representations with strings implanted. 
    Since these are all saved in memory, it is importent to recognize how many representations we can do safely.
    For 10gb assigned, we have ~25 reps. For 25gb available, we have ~60 reps.
    
    '''
    arr = []
    if b_Verbose:
        print("#maps = "+str(num_smaps_per_map))
        print("G_mu = "+str(G_mu))
        print("v = "+str(V))
        print("Amplitude = "+str(Amp))
    for cur_map in base_maps:
        for i in range(num_smaps_per_map):
            _,smap = add_strings(cur_map,G_mu,V,1,None,None,Amp) #string map that is not saved to file
            val = (smap,1)
            if b_Verbose:
                print(type(val))
            arr.append(val)
    arr = np.array(arr)
    if b_Verbose:
        print("Amount of maps in array: "+str(len(arr)))
    return arr


def create_map_array(base_maps,num_smaps_per_map,G_mu,V,Amp,percentage_string = 0.5,b_Verbose = False):
    '''
    Creates an array of maps and indices(2 arrays in one)
    '''
    np.random.shuffle(base_maps)    #array now shuffled
    max_i = int(percentage_string* len(base_maps))-1
    if b_Verbose:
        print("Max_i = "+str(max_i))
    string_maps = base_maps[0:max_i]
    staying_maps = base_maps[max_i:]
    string_maps = create_string_maps_arr(string_maps,num_smaps_per_map,G_mu,V,Amp,b_Verbose)
    if b_Verbose:
        print(type(staying_maps))
        print(type(string_maps))
    
    map_arr = []
    idx_arr = []
    for i_map in staying_maps:
        map_arr.append(i_map)
        idx_arr.append(0)
    #join stacks
    if b_Verbose:
        print("Appending String maps")
    for arra in string_maps:
        map_arr.append(arra[0])
        idx_arr.append(arra[1])
    return [np.array(map_arr),np.array(idx_arr)]
def arr_to_batches(data,batchsize,b_Verbose = False):
    '''
    Turn vector into vector of vectors with batchsize
    '''
    if len(data) <= batchsize:
        return [data]
    overflow = len(data)%batchsize
    max_val = len(data)-overflow
    num_batches = max_val/batchsize
    if b_Verbose:
        print("Off by "+str(overflow))
        print("Maximum value = "+str(max_val))
        print("Generating "+str(num_batches) +" batches.")
    arr = []
    for i in range(int(num_batches)):
        low_lim = (i)*batchsize
        up_lim = (i+1)*batchsize
        arr.append(data[low_lim:up_lim])
    if b_Verbose:
        print("Generated "+str(len(arr)) +"batches.")
        print("Fits with original estimate: "+str(len(arr)==num_batches))
    return arr
    
def turn_class_mat_to_vec(results,use_cuda = False):
    '''
    take a [batchsize,num_classes] matrix, and turn it into a vector signifying wether or not theres strings involved
    '''
    out = []
    for classif in results:
        out.append(np.argmax(classif.data))
    return Variable(torch.from_numpy(np.array(out))).cuda() if use_cuda else Variable(torch.from_numpy(np.array(out)))

def log(s):
    
    if not isinstance(s,str):
        s = str(s)  #cast into string
    with open("log_"+str(datetime.today().day)+"_"+str(datetime.today().month)+"_"+str(datetime.today().year)+".txt","a") as f:
        f.write("["+str(datetime.now())+"]: "+s+"\n")
        print("["+str(datetime.now())+"]: "+s)

if __name__ == "__main__":
    t_start = datetime.now()
    data = load_data(load_filenames(datapath,"sub"))
    print(len(data))
    
    norm_dat = normalize_data(data)
    
    t_elapsed = datetime.now() - t_start
    
    print("Elapsed time one file loading: "+str(t_elapsed))
    dataarr_to_tensor_stack(norm_dat,False)
    