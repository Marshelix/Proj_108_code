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
    x = np.cos(theta)
    print(l)
    print(m)
    
    df = pd.DataFrame(0,index = index, columns = cols)
    print(datapath_leg+filename)
    if not os.path.exists(datapath_leg+filename) or os.path.getsize(datapath_leg+filename) == 0:
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
                

theta = np.linspace(0,2*np.pi,3000)
start = datetime.now()
for l in range(1447,2001):
    for m in range(-l,l+1):
        Plm_save(l,m,theta)
elapsed = datetime.now() - start
print("Elapsed time: "+str(elapsed))