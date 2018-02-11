# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 11:07:45 2018

@author: Martin Sanner

Program to work out spherical harmonics, save them into a pandas dataframe, then save this to a file
Spherical harmonics are generated for modes l0 <= l < l1, supplied by the user
"""

print("This program is meant to calculate the spherical harmonics for l-values between l0,l1-1.")
print("Both l0,l1 are supplied by either a program or the user to begin with. Spherical harmonics are generated using scipy")
print("The data is loaded into a dataframe, which is then saved using pickle.")

import pandas as pd
import pickle
import scipy.special as f_base
import math
import numpy as np
import time
import os
import gc
import psutil
import sys
syspath = os.path.abspath(os.path.join(os.path.dirname(__file__),".."+"/py_mail"))
sys.path.append(syspath)
from mailbot import email_bot
start_time = time.time()

pid = os.getpid()
psu = psutil.Process(pid)
l0 = int(input("l0 = "))
l1 = int(input("l1 = "))

ang_res = 0.01  #resolution of the angle. Number of points N = 2pi/ang_res for both cases

def calculate_harmonics(l0,l1):
    '''
    calculate harmonics using scipy.special.sph_harm between l0,l1-1
    '''
    if l0 == l1:
        print("Error: l0 = l1! Please choose differently!")
        
        l0 = int(input("l0 = "))
        l1 = int(input("l1 = "))
        return calculate_harmonics(l0,l1)
   
    N = int(2*math.pi/ang_res)
    thetas = np.linspace(0,2*math.pi,N)
    phis = np.linspace(0,math.pi,N)
    
    
    for l in range(l0,l1):
        print("===============")
        print("l = " +str(l))
        l_arr = []
        for m in range(-l,l+1):
   
            m_frame = pd.DataFrame(0,index = thetas,columns =[str(x) for x in phis] )
            
            '''
            m_frame = 
                  \Phis| ---------------------------
            Thetas     | ylm
            ___________|
            +
            +
            +
            +
            ...
            '''
            
            func = f_base.sph_harm(l,m,thetas,phis)
            #func is the matrix of values for lm for all thetas, phis
            m_frame.set_value(thetas,str(phis),func)    #casted into m_frame
            l_arr.append(m_frame)
            

        #note: replace path with profile on machine! Needs to be done by user!
        path = "F:\Programming\Project\git\Code\Python\spherical_harmonics_generator\harmonics/"
        file = open(path + "spher_har_l_"+str(l)+".dat","wb")
       
        pickle.dump(l_arr,file)
        print("RAM usage before closing the file: "+str(psu.memory_info()[0]/2**30) + " Gb")
        file.close()
        print("RAM usage before collection: "+str(psu.memory_info()[0]/2**30) + " Gb")
        gc.collect()
        print("RAM usage after collection: "+str(psu.memory_info()[0]/2**30) + " Gb")
        print("File "+path + "spher_har_l"+str(l)+".dat"+" generated and saved.")
    
    return 0

                    
calculate_harmonics(l0,l1)
end_time = time.time()
elapsed = end_time - start_time
print("<===========================>")
print(str(elapsed) + " s = "+str(elapsed/60) +" min = "+str(elapsed/3600) +" h = "+str(elapsed/(3600*24)) +" days passed")