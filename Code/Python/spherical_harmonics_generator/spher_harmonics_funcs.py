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

start_time = time.time()

settings = setup.load_settings()
print(settings)
datapath = settings["Spherharg"][2]  #spherical harmonics data
datapath_leg = settings["Spherharg"][3]

def Plm_save(l,m,theta):
    '''
    theta : Angles to compute the polynomials for
    l: degree of the polynomial
    m: order of the polynomial    
    '''
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
    '''
    Load the legendre values from file
    '''
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
'''
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
'''

known_vals = []
def check_finiteness(l,m,theta):
    '''
    Check the file for the Plm (l,m)
    if data is not finite everywhere, print l,m
    '''
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
'''
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
'''
import pyshtools as sht

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

def load_spectra_from_file(filename,T0 = 2.725):
    '''
    Load a .dat file as a csv, delimiter = "    "
    '''
    df = pd.read_csv(filename,header = None, names = ["TT","EE","TE","UNKNOWN_1","UNKNOWN_2"], index_col = 0,delim_whitespace = True)
    print("Data loaded")
    premul = T0**2*2*np.pi/1
    print(df.head())
    return df["TT"]
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

        