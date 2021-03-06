# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 14:39:17 2018

@author: Martin Sanner

Function file for a custom implementation of spherical harmonics.
This file covers:
    Plm_save(l,x):
        define the Legendre polynomial of the lth degree for x, then save them to the file
    spher_harm_custom(l,m,theta,phi):
        Returns the spherical harmonics over arrays theta, phi for the value of l,m. Returns as a dataframe.
        This functions uses the scipy implementation of the legendre polynomials.
    spher_harm_load(l,m,theta,phi):
        Returns the spherical harmonics over arrays theta, phi for l,m as dataframe.
        Scans the datapath for legendre polynomials, if it can find them, loads the value, if not, generates the file.        
    
"""

import pandas as pd
import os
import pickle
import time
from datetime import datetime
import scipy.special as func_base
import sys
import numpy as np
import psutil
import platform
cur_os = platform.system()
print("OS detected: "+cur_os)
mailpath,setuppath = " "," "
if cur_os == "Windows":
    mailpath = os.path.abspath(os.path.join(os.path.dirname(__file__),".."+"/py_mail"))
    setuppath = os.path.abspath(os.path.join(os.path.dirname(__file__),".."+"/Setup"))
elif cur_os == "Linux":
    mailpath = "../py_mail"
    setuppath = "../Setup"
sys.path.append(mailpath)
sys.path.append(setuppath)
from mailbot import email_bot
from Setup import setup
import scipy.misc as misc
import random
import matplotlib.pyplot as plt
import pyshtools as sht # Spherical Harmonics tools
#Setup
if __name__ == "__main__":
    start_time = time.time()

    settings = setup.load_settings()
    #print(settings)
    datapath = settings["Spherharg"][2]  #spherical harmonics data
    datapath_leg = settings["Spherharg"][3]
    
    #Email bot
    
    username = settings["Email"][4]
    password = settings["Email"][3]
    server = settings["Email"][2]
    
    bot = email_bot(settings["Email"][0],settings["Email"][1],server,password,username,int(settings["Email"][5]))
def load_spectra_from_file(filename,T0 = 2.725):
    ''' 
    Load a .dat file as a csv, delimiter = "    "
    '''
    df = pd.read_csv(filename,header = None, names = ["TT","EE","TE","UNKNOWN_1","UNKNOWN_2"], index_col = 0,delim_whitespace = True)
    print("Spectrum loaded from "+filename)
    '''
    start with T0^2 l(l+1)/2pi Cl
    want Cl/(2l+1)
    multiply 1 by 2pi/(T0^2*(l(l+1)(2l+1))
    '''
    l = np.array(df.index)
    premul = 10**-12/(l*(l+1))#1#T0**-2*2*np.pi
    '''
    print(type(premul))
    print(len(premul))
    print(len(df.index))
    print(premul)
    print(df.head())
    '''

    return df["TT"]*premul
if __name__ == "__main__":
    gen_data_path = settings["Spherharg"][4]
filename = "G:\Data\camb_74022160_scalcls.dat"
info_name = "G:\Data\Data_camb_74022160.txt" #data on what was used to compute it

def read_info(info_name):
    '''
    read_info: Opens a txt file containing the information used to generate the input map vector.
    argument:
    info_name: str: path + filename to the location
    '''
    with open(info_name,"r") as f:
        for line in f.readlines():
            print(line)

num_maps = 0

def gen_map(power_filename, info_name,T0 = 2.725,num_maps = 0,mode = "DH2"):
    '''
     Generate a Temperature map using SHtools.
     Saves this map to a file, named the same as the original and info .png.
    '''
    #find filename for saving
    img_name = gen_data_path+"maps/"
    for c in power_filename:
        if c not in gen_data_path:
            if c is ".":
                break
            else:
                img_name = img_name + c
    
    #filename found
    file_ext_dat = "_"+str(num_maps)+"_data.png"
    file_ext_dat_bar = "_bar"+file_ext_dat
    #file_ext_grad = "_"+str(num_maps)+"_grad.png"
    #file_ext_norm = "_"+str(num_maps)+"_grad_norm.png"
    TT_spectrum = load_spectra_from_file(power_filename)
    #TT spectrum works as intended
    clm = sht.SHCoeffs.from_random(TT_spectrum,normalization = "ortho",csphase = -1)
     
    grid = clm.expand(grid = mode) #T map
    cur_map = grid.to_array()

    cur_map = cur_map/T0
    grid = grid/T0
    print("max T: "+str(np.max(cur_map)))
    
    dat_img = img_name + file_ext_dat
    dat_img_bar = img_name + file_ext_dat_bar
    
    fig,ax = plt.subplots()
    cax = ax.imshow(cur_map)
    fig.colorbar(cax)
    grid.plot()#grid.plot(fname = dat_img)
    #ax.set_title("T_max = "+str(np.max(grid.data)))
    #print("Saving data to "+dat_img_bar)
    #fig.savefig(dat_img_bar)
    
    '''
    grad_data = np.array(np.gradient(grid.data))[1]/T0
    grid2 = sht.SHGrid.from_array(grad_data) #dT/T map
    grad_img = img_name + file_ext_grad
    print("Saving gradient to "+grad_img)
    fig,ax = grid2.plot(fname = grad_img)

    #no normalization required at this point
    norm_img = img_name+ file_ext_norm
    print(norm_img)
    maxmin = (np.max(grid2.data)-np.min(grid2.data))
    grid2.data = (grid2.data - np.min(grid2.data))/maxmin
    fig,ax = grid2.plot(fname = norm_img)
    '''
    #apparently, grid turns into a tuple for some reason
    #Force grid to stay DHRealGrid
    return sht.SHGrid.from_array(cur_map)



def add_strings(grid, G_mu,v,num_strings,tgname,sname,A = 0,b_Verbose = False):
    '''
        Add n strings to a 0 grid, with random directions.
        Strings are assumed to be 1 element thin, ie only in one dimension, and work as a step function
        
        Positions are chosen from the dimensions of the grid, ie x,y = rand()%grid.data.shape[0],rand()%grid.data.shape[1]
        Strings are generated starting at position (x,y) in direction dir = rand()%4
        
        1 2 3
        4 X 4
        3 2 1
        
        
    '''
    gamma = 1
    if v is not 1:
        gamma = 1/np.sqrt(1-v**2)
    amp = 0
    if A != 0:
        amp = 8*np.pi*A*gamma
    else:
        amp = 8*np.pi*v*G_mu*gamma
    if A != G_mu*v:
        v = A/G_mu#force velocity to fit for now
    grid_dim = grid.data.shape
    
    xmax = grid_dim[0]
    ymax = grid_dim[1]
    xmid = xmax/2
    ymid = ymax/2
    rad = int(0.1*xmax)#10% max size
    new_grid_data = np.zeros(shape = (xmax,ymax))
    if b_Verbose:
        print("Amplitude: "+str(amp))
    for i in range(num_strings):
        '''
            For each string: 
                choose random pos
                choose random direction
                step = amp
                
        '''
        #pick direction
        #Strings have to be centered at the moment for testing
        dire = random.randint(1,4)
        if dire  == 1:
            xi = random.randint(xmid-rad,xmid+rad)
            yi = random.randint(ymid-rad,ymid+rad)
            b = yi+xi
            
            for x in range(0,xmax):
                for y in range(0,ymax):
                    new_grid_data[x][y] =new_grid_data[x][y]+ amp*((int(y >= -x+b)-0.5))#step on line
        elif dire == 2:
            xi = random.randint(xmid-rad,xmid+rad)
            
            for x in range(0,xmax):
                new_grid_data[x][:] =new_grid_data[x][:]+ amp*((int(x >= xi)-0.5))
        elif dire == 3:
            xi = random.randint(xmid-rad,xmid+rad)
            yi = random.randint(ymid-rad,ymid+rad)
            b = yi-xi
            for x in range(0,xmax):
                for y in range(0,ymax):
                    new_grid_data[x][y] =new_grid_data[x][y]+ amp*((int(y >= x+b)-0.5))#step on line
        elif dire == 4:
            yi = random.randint(ymid-rad,ymid+rad)
            for y in range(0,ymax):
                new_grid_data[:][y] = new_grid_data[:][y]+ amp*((int(y >= yi)-0.5))
        
            
    new_map = sht.SHGrid.from_array(new_grid_data)
    if sname is not None:
        f,ax = new_map.plot(fname = sname)
    else:
        f,ax = new_map.plot()
    ax.set_title("New map G_mu = "+str(G_mu)+",A = "+str(A))
    total_grid_data = new_grid_data + grid.data
    total_grid = sht.SHGrid.from_array(total_grid_data)
    if b_Verbose:
        print("TG Data size: "+str(total_grid_data.shape))
    if tgname is not None:
        f2,ax2 = total_grid.plot(fname = tgname)
    else:
        f2,ax2 = total_grid.plot()
    ax2.set_title("Total map - T_Min = "+str(np.min(total_grid.data))+" - T_Max = "+str(np.max(total_grid.data)))
    plt.close("all")
    return new_map,total_grid

#g1 = gen_map(filename,info_name)

#nm,tg = add_strings(g2,20,10**-4,2000,10)
'''
print("T_Max in String Map: "+str(np.max(nm.data)))
print("T_Max in total Map: "+str(np.max(tg.data)))
print("T_Max in original map: "+str(np.max(g1.data)/2.725))
'''
def generate_maps(num_files,num_file_start = 0,b_hasStrings = True,f_stringPercentage = 0.7,f_T0 = 2.725,s_pfilename = filename,s_infofilename = info_name,b_verbose = False):
    '''
    
    NOT IN USE
    
    
        Generate n maps, save them to a file labelled based on its own name and the number of files already put in.
        Arguments:
            num_files: Integer: forced: Number of maps you want to generate.
            num_file_start: Integer: Standard 0
                            starts map generation at this point
            b_hasStrings: Standard True
                          Wether this iteration includes files with strings. 
                          These files are producing more data. 
                          If no string is included, only two files is generated.
                          If strings are included(regardless of count), we shall also generate the string map and the total map.
            f_stringPercentage: Standard 0.7
                                Only active if b_hasStrings == True
                                Percentage of files with strings to be generated -> n_file_strings = floor(f_stringPercentage*num_files)
            f_T0: Standard 2.725
                  Average temperature of the CMB, rescales dT -> dT/T, the map we get from gen_map()
            s_pfilename:Standard only available power file 
                        filename for the powers to be passed into the gen_map function. 
            s_infofilename: Standard only available info file
                        filename for additional info on the data used for s_pfilename
            b_verbose: Standard False
                       Whether or not to have additional output
    '''
    t_start = datetime.now()
    #send an email
    
    bot.set_topic("Started to generate "+str(num_files)+" maps")
    bot.append_message("Program has commenced at " + str(t_start))
    
    #for psutil
    pid = os.getpid()
    psu = psutil.Process(pid)
    if num_file_start < 0:
        num_file_start = np.floor(-num_file_start)
    if num_file_start  != int(num_file_start):
        num_file_start = int(num_file_start)

    if num_files != np.floor(num_files):
        num_files = np.floor(num_files)
    num_stringmaps = 0
    if b_hasStrings:
        num_stringmaps = np.floor(num_files*f_stringPercentage)
    if b_verbose:
        read_info(s_infofilename)
    
    error_counter = 0
    for i in range(num_file_start,num_file_start +num_files):
        g = gen_map(s_pfilename,s_infofilename,f_T0,i)
        if i < num_stringmaps:
            #generate additional strings
            #find a way to have G_mu, v be a range
            n_strings = 1
            G_mu = 10**-5   #c = 1
            v = 0.01 #c = 1
            A = v*G_mu
            #String map name
            img_name = gen_data_path+"maps/"
            for c in s_pfilename:
                if c not in gen_data_path:
                    if c is ".":
                        break
                    else:
                        img_name = img_name + c
            smname = img_name + "_stringmap"+str(i)+".png"
            tgname = img_name+"_totalmap"+str(i)+".png"
            print("Saving string map to "+smname)
            print("Saving total map to "+tgname)
            string_map,total_grid = add_strings(g,G_mu,v,n_strings,tgname,smname,A)
        #send mail if too much harddrive is used up
        if (psutil.cpu_percent()) > 95:
            bot.set_topic("Error occured")
            bot.append_message("["+str(datetime.now())+"]: "+"Too high cpu usage repeatedly at i =" + str(i),False)
            print("["+str(datetime.now())+"]: "+"Too high cpu usage repeatedly at i =" + str(i))
            error_counter = error_counter+1
        if (psu.memory_info()[0]/2**30)*100/16 > 95:
            #Ram keeps going up steadily, but slowly. Stop now.
            bot.set_topic("Error occured")
            bot.append_message("["+str(datetime.now())+"]: "+"Too much ram space used at i =" + str(i),False)
            print("["+str(datetime.now())+"]: "+"Too much ram space used at i =" + str(i))
            error_counter = error_counter+1
        if psutil.disk_usage(gen_data_path)[3] > 98:
            #no more disk space for next file: Stop calculation.
            bot.set_topic("Error occured")
            bot.append_message("["+str(datetime.now())+"]: "+"Too much harddrive space used at i =" + str(i))
            print("["+str(datetime.now())+"]: "+"Too much harddrive space used at i =" + str(i))
            return 0
        if error_counter > 3:
            bot.set_topic("Errors occured")
            bot.append_message("["+str(datetime.now())+"]: "+"Closing Program early at i = "+str(i))
            return 0
        plt.close("all")
    t_elapsed = datetime.now() - t_start
    print("Elapsed time = "+str(t_elapsed))
    bot.set_topic("Map generation Finished")
    bot.append_message("Program has finished at " + str(datetime.now()),False)
    bot.append_message("Elapsed time: "+str(t_elapsed)) #message sent
    return 0

def find_map_id(tw,pw,ang_size,max_angle = 360):
    '''
    
    NOT IN USE
    
    Given an array of all maps of a given realization, return the id in the vector for the map in which a certain theta,phi lies
    
    '''
    num_maps_line = max_angle/ang_size
    idx = 0
    horizontal_part = tw/ang_size - (tw%ang_size)/ang_size
    pws = 90-pw
    vertical = pws/ang_size - (pws%ang_size)/ang_size
    
    print("vertical: "+str(vertical))
    print("horizontal: "+str(horizontal_part))
    idx = horizontal_part + vertical*num_maps_line
    print("Index: "+str(idx))
    return int(idx)

def generate_map_degrees(power_file,info_file = "",T0 = 2.725,num_map = 0,angular_size = 10,b_Verbose = False):
    '''
    
    Can read out the 10x10 map more or less, slightly buggy, not in use for this reason.
    
    
    Function to generate an LCDM map and split it up into 10°x10° patches, saving all of these to disk.
    Generates one map and split it up into the maps.
    
    Arguments:
        power_file: String: filename for the power to be used.
        info_file: String: Standard = ""
                            filename for the info file associated with the power file used.
        T0:float: Standard = 2.725
                            Background Temperature of the Universe, to normalize the dT/T map
        num_map:int: Standard = 0:
                            maps are saved to a file with moniker _(num_map)_(num_patch)
        angular_size: int: Standard = 10:
                            The size the map is split up into
        b_Verbose: bool: Standard False:
                            Whether or not to print out extra information
    Returns:
        all_maps: Array:
            all_maps is the array of all grids calculated for this method, ie the original full map and all segments from it. Each element is a sht.SHGrid
    '''
    if info_file is not "":
        if b_Verbose:
            read_info(info_file)
    if type(angular_size) is not type(int):
        angular_size = int(angular_size)
    if type(num_map) is not type(int):
        num_map = int(num_map)
     #find filename for saving
    img_name = gen_data_path+"maps/"
    for c in power_file:
        if c not in gen_data_path:
            if c is ".":
                break
            else:
                img_name = img_name + c
    img_name = img_name + "_"+str(num_map)
    TT_spectrum = load_spectra_from_file(power_file)
    #TT spectrum works as intended
    clm = sht.SHCoeffs.from_random(TT_spectrum,normalization = "ortho",csphase = -1)
    gridfile_name = img_name + ".png"
    grid = clm.expand() #T map
    
    grid = grid/T0
    if b_Verbose:
        print("max T: "+str(np.max(grid.data)))
        print(type(grid))
    f0,ax0 = grid.plot(fname = gridfile_name)
    plt.close(f0)
    num_plots_lat = int(360/angular_size)
    num_plots_long = num_plots_lat#int(360/angular_size)
    lat_range = int(grid.data.shape[1]/num_plots_lat)
    long_range = int(grid.data.shape[0]/num_plots_long)
    if b_Verbose:
        print(num_plots_lat,num_plots_long,lat_range,long_range)
    subplot = 0
    all_maps = [grid]
    
    for a in range(1,num_plots_lat+1):
        for b in range(1,num_plots_long+1):
            #
            #   NOTE: There is a bug with the software where A: sht implies N_lat %2 = 0 and N_long = N_lat or N_long = 2*Nlat
            #   Including an overlapping region fixes this issue
            #
            subplot = subplot + 1
            plt.close("All")
            min_lat = (a-1)*lat_range
            max_lat = (a*lat_range) +1   #Include overlapping region
            range1 = np.array(range(min_lat,max_lat))
            min_long = (b-1)*long_range
            max_long = b*long_range+1   #include overlapping region
            range2 = np.array(range(min_long,max_long))
            if b_Verbose:
                print("Latitude between ["+str(np.min(range1))+","+str(np.max(range1))+"]")
                print("Longitude between ["+str(np.min(range2))+","+str(np.max(range2))+"]")
            g1_ext = grid.data[range2]
            #g1_ext.shape = (222,3998)
            #need to extract out thecorrect longitude
            g2 = []
            for i in range(g1_ext.shape[0]):
                g2.append(g1_ext[i][range1])
                #now g2 includes all vecs in the proper range
            g2 = np.array(g2)
            if b_Verbose:
                print("G2 shape: "+str(g2.shape))
            g3 = sht.SHGrid.from_array(g2)
            filename = img_name + "_"+str(subplot)+".png"
            if b_Verbose:
                print("Plot to be saved to "+filename)
            #f,ax = g3.plot()
            #todo: implement colorbar etc
            
             
            f,ax = plt.subplots()
            cax = ax.imshow(g3.data)
            f.colorbar(cax)
            
            f.savefig(filename)
            plt.close(f)
            plt.close("All")
            all_maps.append(g3)
    #save all_maps to file
    amf_name = img_name + ".dat"
    f = open(amf_name,"wb")
    pickle.dump(all_maps,f)
    return all_maps
    
#am = generate_map_degrees(filename,num_map = 2,b_Verbose = True)
#save am to file

def get_sub_maps(am,tws,pws,ang_res):
    '''
    Return the array of maps  with certain angles
    
    tws,pws: array of ints or ints
    am: array of all maps
    ang_res: angular resolution of maps
    
    NOT IN USE
    '''
    arr = []
    print(type(tws))
    print(type(pws))
    if type(tws) != type(np.linspace(0,1)):
        if type(pws) != type(np.linspace(0,1)):
            #only one value
            
            return np.array([am[find_map_id(tws,pws,ang_res)]])
        else:
            for pw in pws:
                arr.append(am[find_map_id(tws,pw,ang_res)])
    if type(pws) != type(np.linspace(0,1)):
        #tws is an array, pws is only one value
        if type(pws) == int:
            for tw in tws:
                arr.append(am[find_map_id(tw,pws,ang_res)])
        else:
            print("pws must be either an int or an array of ints!")
            return np.array([])
    elif type(pws) == type(np.linspace(0,1)) and type(tws) == type(np.linspace(0,1)):
        for pw in pws:
            for tw in tws:
                arr.append(am[find_map_id(tw,pw,ang_res)])
    return np.array(arr)
#pull out maps on 
#works: sub_maps = get_sub_maps(am,np.linspace(0,350,10),0,10)

def generate_multimap_subset(pfile,ifile,map_ids,T0 = 2.725,angular_size = 10,b_Verbose = False,pws = 0,tws = np.linspace(0,350,20)):
    '''
    Generate a set of subset of maps with given angular data.
    Automatically assume equator
    map_ids = array of ids for the map generator
    
    GENERATES TOO MUCH DATA =========== IGNORE
    '''
    pid = os.getpid()
    psu = psutil.Process(pid)
    arr = []
    originals = []
    for idx in map_ids:
        ami = generate_map_degrees(pfile,ifile,T0,idx,angular_size,b_Verbose)
        originals.append(ami[0])
        sub = get_sub_maps(ami,tws,pws,angular_size)
        arr.append(sub)
        error_counter = 0
        if (psutil.cpu_percent()) > 95:
            bot.set_topic("Error occured")
            bot.append_message("["+str(datetime.now())+"]: "+"Too high cpu usage repeatedly at idx =" + str(idx),False)
            print("["+str(datetime.now())+"]: "+"Too high cpu usage repeatedly at idx =" + str(idx))
            error_counter = error_counter+1
        if (psu.memory_info()[0]/2**30)*100/16 > 95:
            #Ram keeps going up steadily, but slowly. Stop now.
            bot.set_topic("Error occured")
            bot.append_message("["+str(datetime.now())+"]: "+"Too much ram space used at idx =" + str(idx),False)
            print("["+str(datetime.now())+"]: "+"Too much ram space used at idx =" + str(idx))
            error_counter = error_counter+1
        if psutil.disk_usage(gen_data_path)[3] > 98:
            #no more disk space for next file: Stop calculation.
            bot.set_topic("Error occured")
            bot.append_message("["+str(datetime.now())+"]: "+"Too much harddrive space used at idx =" + str(idx))
            print("["+str(datetime.now())+"]: "+"Too much harddrive space used at idx =" + str(idx))
            return 0
        if error_counter > 3:
            bot.set_topic("Errors occured")
            bot.append_message("["+str(datetime.now())+"]: "+"Closing Program early at idx = "+str(idx))
            return 0
    return np.array(originals),np.array(arr)
if __name__ == "__main__":
    t_start = datetime.now()
    plt.close("all")
'''
v = 0.5
G_mu = 10**-6
A = v*G_mu

tgname = "tgrid.png"
sname = "sgrid.png"
test_map = gen_map(filename,info_name)
submap_arr_slice = test_map.data[1750:2250]
submap_arr = []
for vi in submap_arr_slice:
    arr = []
    for i in range(500):
        arr.append(vi[i])
    arr = np.array(arr)
    submap_arr.append(arr)
submap_arr = np.array(submap_arr)
print("submap min,max: ("+str(np.min(submap_arr))+","+str(np.max(submap_arr))+")")
print(submap_arr.shape)
submap = sht.SHGrid.from_array(submap_arr)
submap.plot()
plt.title("min,max: ("+str(np.min(submap_arr))+","+str(np.max(submap_arr))+")")

sm,tg = add_strings(submap,G_mu,v,1,tgname,sname,A)
print("Sm min,max: ("+str(np.min(sm.data))+","+str(np.max(sm.data))+")")
print("Tg min,max: ("+str(np.min(tg.data))+","+str(np.max(tg.data))+")")

print(np.min(submap_arr)-np.min(tg.data),np.max(submap_arr)-np.max(tg.data))
'''
def log(s):
    
    if not isinstance(s,str):
        s = str(s)  #cast into string
    with open("log_"+str(datetime.today().day)+"_"+str(datetime.today().month)+"_"+str(datetime.today().year)+".txt","a") as f:
        f.write("["+str(datetime.now())+"]: "+s+"\n")
        print("["+str(datetime.now())+"]: "+s)



def gen_multiple_maps(n_maps,filename,info_name,G_mu = 10**-6,v = 0.5, b_Verbose = True,n_min = 0,save_freq = 10):
    '''
    Generate multiple maps, save them to a file
    
    CURRENT MAP GENERATION PROCESS
    '''
    log("Generating "+str(n_maps)+" maps.")
    A = G_mu*v
    if b_Verbose:
        log("#maps= "+str(n_maps))
        log("Filename = "+str(filename))
        log("Info file = "+ str(info_name))
        log("G_mu = "+str(G_mu))
        log("V = "+str(v))
    if n_maps < 0:
        log("n_maps < 0. Inverting")
        n_maps = abs(n_maps)
    sub_maps = []
    
    full_maps = []
    string_maps = []
    total_maps = []
    old_save = n_min
    for i in range(n_min,n_min + n_maps+1):
        log("Creating map #"+str(i))
        cur_map_full = gen_map(filename,info_name,2.7255,i)
        full_maps.append(cur_map_full)
        submap_arr_slice = cur_map_full.data[1889:2111] #10 degrees: total sidewidth: 222
        submap_arr = []
        for vi in submap_arr_slice:
            arr = []
            for k in range(submap_arr_slice.shape[0]):
                arr.append(vi[k])
            arr = np.array(arr)
            submap_arr.append(arr)
        #generate tgname, sname
        img_name = "G:\Data\maps/"
        for c in filename:
            if c not in gen_data_path:
                if c is ".":
                    break
                else:
                    img_name = img_name + c
        img_name = img_name + "_"+str(i)
        tgname = img_name + "_grid.png"
        sname = img_name + "_strings.png"
        #log("Tgname = "+tgname)
        #log("Sname = "+sname)
        submap_arr = np.array(submap_arr)
        #log("submap min,max: ("+str(np.min(submap_arr))+","+str(np.max(submap_arr))+")")
        #log(submap_arr.shape)
        submap = sht.SHGrid.from_array(submap_arr)
        submap.plot()
        sub_maps.append(submap)
        plt.title("min,max: ("+str(np.min(submap_arr))+","+str(np.max(submap_arr))+")")
        dat_img = img_name + "_grid_bar.png"
        fig,ax = plt.subplots()
        cax = ax.imshow(submap.data)
        fig.colorbar(cax)
        ax.set_title("T_max = "+str(np.max(submap.data)))
        #log("Saving data to "+dat_img)
        
        fig.savefig(dat_img)
        sm,tg = add_strings(submap,G_mu,v,1,tgname,sname,A)
        #log("Sm min,max: ("+str(np.min(sm.data))+","+str(np.max(sm.data))+")")
        #log("Tg min,max: ("+str(np.min(tg.data))+","+str(np.max(tg.data))+")")
        string_maps.append(sm)
        total_maps.append(tg)
        log(np.min(submap_arr)-np.min(tg.data))
        log(np.max(submap_arr)-np.max(tg.data))
        plt.close("all")
        
        #save if save_freq reached
        if (i-old_save) % save_freq == 0:
            log("Saving to files.")
            tmname = img_name +"__"+str(old_save)+"_"+str(i)+ "_tmaps.dat"
            smfname = img_name+"__"+str(old_save)+"_"+str(i)+"_smaps.dat"
            fmname = img_name +"__"+str(old_save)+"_"+str(i)+ "_fmaps.dat"
            sbname = img_name +"__"+str(old_save)+"_"+str(i)+"_sub.dat"
            log(tmname+","+smfname+","+fmname+","+sbname)
            with open(tmname,"wb") as f:
                pickle.dump(np.array(total_maps),f)
            with open(smfname,"wb") as f:
                pickle.dump(np.array(string_maps),f)
            with open(fmname,"wb") as f:
                pickle.dump(np.array(full_maps),f)
            with open(sbname,"wb") as f:
                pickle.dump(np.array(sub_maps),f)
            total_maps = []
            
            string_maps = []
            full_maps = []
            sub_maps = []
            old_save = i
            log("Files saved")
            
    return np.array(total_maps),np.array(string_maps),np.array(full_maps),np.array(sub_maps)
if __name__ == "__main__":
    t_start = datetime.now()
    gen_multiple_maps(200,filename,info_name,n_min = 1040)
    t_elapsed = datetime.now() - t_start
    print("Elapsed time = "+str(t_elapsed))

'''

#origs,arr = generate_multimap_subset(filename,"",range(3,20),b_Verbose = True)
#with open("origins.dat","wb") as f1:
#    pickle.dump(origs,f1)
#with open("arr.dat","wb") as f2:
#    pickle.dump(arr,f2)
bot.set_topic("Starting Program")
bot.append_message("["+str(datetime.now())+"]: "+"Starting Program")
t_elapsed = datetime.now() - t_start
print("Elapsed time = "+str(t_elapsed))
bot.set_topic("Program finished")
bot.append_message("["+str(datetime.now())+"]: "+"Closing Program")
'''