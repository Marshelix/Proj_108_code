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

mailpath = os.path.abspath(os.path.join(os.path.dirname(__file__),".."+"/py_mail"))
sys.path.append(mailpath)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),".."+"/Setup")))
from mailbot import email_bot
from Setup import setup
import scipy.misc as misc
import random
import matplotlib.pyplot as plt
import pyshtools as sht # Spherical Harmonics tools
#Setup

start_time = time.time()

settings = setup.load_settings()
print(settings)
datapath = settings["Spherharg"][2]  #spherical harmonics data
datapath_leg = settings["Spherharg"][3]

#Email bot

username = settings["Email"][4]
password = settings["Email"][3]
server = settings["Email"][2]

bot = email_bot(settings["Email"][0],settings["Email"][1],server,password,username,int(settings["Email"][5]))

'''

def Plm_save(l,m,theta):
    
    SCIPY IMPLEMENTATION
    
    
    theta : Angles to compute the polynomials for
    l: degree of the polynomial
    m: order of the polynomial    


    filename = "leg_l_"+str(l)+"_m_"+str(m)+".dat"
    cols = [str(m)]
    index = theta


    df = pd.DataFrame(0,index = index, columns = cols)
    print(datapath_leg+filename)
    if not os.path.exists(datapath_leg+filename) or os.path.getsize(datapath_leg+filename) == 0:
        x = np.cos(theta)
        gen_time_start = time.time()
        pl_vals = func_base.lpmv(m,l,x)
        gen_time_elapsed = time.time() - gen_time_start
        print("Generation of data took : "+repr(gen_time_elapsed))
        df = pd.DataFrame(pl_vals,index = index, columns = cols)
        file_time_start = time.time()
        file = open(datapath_leg+filename,"wb")
        print("File opened")
        pickle.dump(df,file)
        print("file written to")
        file.close()
        file_time_elapsed = time.time()-file_time_start
        print("File generation took : "+repr(file_time_elapsed))
    else:
         if os.path.getsize(datapath_leg+filename) > 0:
            print("Loading from file.")
            file = open(datapath_leg+filename,"rb")
            df2 = pickle.load(file)
            file.close()
            same_angles_check = len(index) == len(df2.index)
            if same_angles_check:
                for i in range(len(df2.index)):
                    same_angles_check = same_angles_check and df2.index[i] == index[i]
            if same_angles_check:
                print("File loaded. Same angles.")
                df = df2
    print("==================")
    return df
                

theta = np.linspace(0,2*np.pi,300)
from scipy.misc import factorial
def ylm_builtin(l,m,theta,phi):
    prefac = np.sqrt((2*l+1)/2*factorial(l-m)/factorial(l+m))
    print(prefac)
    legend = func_base.lpmv(m,l,np.cos(theta)) # 1*len(theta) array
    print("Legend")
    print(legend)
    exponents = [m*1j*x for x in phi]
    expo = np.exp(exponents)
    print("Expo")
    print(expo)
    values = prefac * np.outer(legend,expo)
    
    cols = [str(x) for x in phi]
    ind = theta
    df = pd.DataFrame(values,index = ind,columns = cols)
    return df

def ylm_load(l,m,theta,phi):

    prefac = np.sqrt((2*l+1)/2*factorial(l-m)/factorial(l+m))
    print(prefac)
    legend = Plm_save(l,m,theta)    #checks if file exists, generates and saves it if not.
    leg_data = legend[str(m)]
    print("Legend load:")
    print(leg_data)
    exponents = [m*1j*x for x in phi]
    expo = np.exp(exponents)
    print("Expo load: ")
    print(expo)
    values = prefac * np.outer(leg_data,expo)
    
    cols = [str(x) for x in phi]
    ind = theta
    df = pd.DataFrame(values,index = ind,columns = cols)
    return df

phi = np.linspace(0,np.pi,3000)
l = m = 85
start_time_builtin = time.time()
df1 = ylm_builtin(l,m,theta,phi)
time_elapsed_builtin = time.time() - start_time_builtin
print(repr(time_elapsed_builtin))
start_time_load = time.time()
df2 = ylm_load(l,m,theta,phi)
time_elapsed_load = time.time() - start_time_load
print(repr(time_elapsed_load))
theta,phi = np.meshgrid(theta,phi)
data = func_base.sph_harm(l,m,theta,phi)
print(type(data))
print(data.shape)

ylm = pd.DataFrame(func_base.sph_harm(l,m,theta,phi),index = theta, columns = [str(x) for x in phi])

known_vals = []
def check_finiteness(l,m,theta):
    
    Check the file for the Plm (l,m)
    if data is not finite everywhere, print l,m
    
    df = Plm_save(l,m,theta)
    known = False
    for colname in df:
        for elem in df[colname]:
            if known:
                return -1
            if not np.isfinite(elem):
                print("(l,m) = ("+str(l)+","+str(m)+")")
                known = True
                known_vals.append((l,m))
                print("==========================")

ls = range(0,501)
filenum = sum([2*l+1 for l in ls])
i = 1
for l in ls:
    for m in range(-l,l+1):
        print(type(l))
        print(type(m))
        check_finiteness(l,m,theta)
        print(str(100*i/filenum)+"%")
        i = i+1
print("Found "+str(len(known_vals))+" values: ")
for elem in known_vals:
    print(elem)
np.savetxt(datapath_leg+"known_vals.txt",known_vals,delimiter = ",")



def Plm_pyshtools_asDF(lmax, theta):
    x = np.cos(theta)
    print(x)
    print(type(x))
    if type(x) == np.ndarray:
        plms = []
        print("element is vector")
        for elem in x:
            plm = sht.legendre.PlmON(lmax,elem)
            plms.append(plm)
        return plms
    else:
        plm = sht.legendre.PlmON(lmax,x)
    print(type(plm))
    print(plm)
    print(plm.shape)
    return plm
'''
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
    l = df.index
    premul = T0**-2*2*np.pi/(l*(l+1))#*(2*l+1)) #keep the 2l+1 factor out to test
    '''
    print(type(premul))
    print(len(premul))
    print(len(df.index))
    print(premul)
    print(df.head())
    '''
    return df["TT"]*premul
gen_data_path = settings["Spherharg"][4]
filename = gen_data_path+"camb_74022160_scalcls.dat"
info_name = gen_data_path+"Data_camb_74022160.txt" #data on what was used to compute it

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

def gen_map(power_filename, info_name,T0 = 2.725,num_maps = 0):
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
     
    grid = clm.expand() #T map
    
    grid = grid/T0
    print("max T: "+str(np.max(grid.data)))
    
    dat_img = img_name + file_ext_dat
    dat_img_bar = img_name + file_ext_dat_bar
    
    fig,ax = plt.subplots()
    cax = ax.imshow(grid.data)
    fig.colorbar(cax)
    grid.plot(fname = dat_img)
    ax.set_title("T_max = "+str(np.max(grid.data)))
    print("Saving data to "+dat_img_bar)
    fig.savefig(dat_img_bar)
    
    
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
    
    #apparently, grid turns into a tuple for some reason
    #Force grid to stay DHRealGrid
    return clm.expand()/T0


def add_strings(grid, G_mu,v,num_strings,tgname,sname,A = 0):
    '''
        Add n strings to a 0 grid, with random directions.
        Strings are assumed to be 1 element thin, ie only in one dimension, and work as a step function
        
        Positions are chosen from the dimensions of the grid, ie x,y = rand()%grid.data.shape[0],rand()%grid.data.shape[1]
        Strings are generated starting at position (x,y) in direction dir = rand()%4
        
        1 2 3
        4 X 4
        3 2 1
        
        
    '''
    amp = 0
    if A != 0:
        amp = 8*np.pi*A
    else:
        amp = 8*np.pi*v*G_mu
    if A != G_mu*v:
        v = A/G_mu#force velocity to fit for now
    grid_dim = grid.data.shape
    xmax = grid_dim[0]
    ymax = grid_dim[1]

    new_grid_data = np.zeros(shape = (xmax,ymax))
    
    print("Amplitude: "+str(amp))
    for i in range(num_strings):
        '''
            For each string: 
                choose random pos
                choose random direction
                step = amp
                
        '''
        #pick direction
        dire = random.randint(1,4)
        if dire  == 1:
            xi = random.randint(0,xmax)
            yi = random.randint(0,ymax)
            b = yi+xi
            
            for x in range(0,xmax):
                for y in range(0,ymax):
                    new_grid_data[x][y] =new_grid_data[x][y]+ amp*((int(y >= -x+b)-0.5))#step on line
        elif dire == 2:
            xi = random.randint(0,xmax)
            
            for x in range(0,xmax):
                new_grid_data[x][:] =new_grid_data[x][:]+ amp*((int(x >= xi)-0.5))
        elif dire == 3:
            xi = random.randint(0,xmax)
            yi = random.randint(0,ymax)
            b = yi-xi
            for x in range(0,xmax):
                for y in range(0,ymax):
                    new_grid_data[x][y] =new_grid_data[x][y]+ amp*((int(y >= x+b)-0.5))#step on line
        elif dire == 4:
            yi = random.randint(0,ymax)
            for y in range(0,ymax):
                new_grid_data[:][y] = new_grid_data[:][y]+ amp*((int(y >= yi)-0.5))
        
            
    new_map = sht.SHGrid.from_array(new_grid_data)
    f,ax = new_map.plot(fname = sname)
    ax.set_title("New map G_mu = "+str(G_mu)+",A = "+str(A))
    total_grid_data = new_grid_data + grid.data
    total_grid = sht.SHGrid.from_array(total_grid_data)
    print("TG Data size: "+str(total_grid_data.shape))
    f2,ax2 = total_grid.plot(fname = tgname)
    ax2.set_title("Total map - T_Min = "+str(np.min(total_grid.data))+" - T_Max = "+str(np.max(total_grid.data)))
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
            max_lat = (a*lat_range)+1   #Include overlapping region
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
t_start = datetime.now()
origs,arr = generate_multimap_subset(filename,"",range(3,1000),b_Verbose = True)
with open("origins.dat","wb") as f1:
    pickle.dump(origs,f1)
with open("arr.dat","wb") as f2:
    pickle.dump(arr,f2)
bot.set_topic("Starting Program")
bot.append_message("["+str(datetime.now())+"]: "+"Starting Program")
t_elapsed = datetime.now() - t_start
print("Elapsed time = "+str(t_elapsed))
bot.set_topic("Program finished")
bot.append_message("["+str(datetime.now())+"]: "+"Closing Program")
