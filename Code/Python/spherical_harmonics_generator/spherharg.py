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
import os                              #filesaving
l0 = int(input("l0 = "))
l1 = int(input("l1 = "))

ang_res = 0.01

def calculate_harmonics(l0,l1):
    '''
    calculate harmonics using scipy.special.sph_harm between l0,l1-1
    '''
    if l0 == l1:
        print("Error: l0 = l1! Please choose differently!")
        
        l0 = int(input("l0 = "))
        l1 = int(input("l1 = "))
        return calculate_harmonics(l0,l1)
    print("Generating vectors")
    N = int(2*math.pi/ang_res)
    thetas = np.linspace(0,2*math.pi,N)
    phis = np.linspace(0,math.pi,N)
    print("Vectors of size "+str(N) +" created.")
    arrays = []
    for l in range(l0,l1):
        print("===============")
        print("l = " +str(l))
        l_arr = []
        for m in range(-l,l+1):
            print("+++++++++++++")
            print("m = "+str(m))
            print("Creating frame")
            m_frame = pd.DataFrame(0,index = thetas,columns =[str(x) for x in phis] )
            #m_frame.set_index(thetas)   #set thetas to be the index.
            print("Frame "+str(m) +" created.")
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
            for theta in thetas:
                for phi in phis:
                    func = f_base.sph_harm(l,m,theta,phi)
                    #bug: For eg m = -1, func is NaN, which m_frame complains about and breaks.
                    m_frame.set_value(theta,str(phi),func)
            l_arr.append(m_frame)
            
            print("+++++++++++++")    
        arrays.append(l_arr)
        #note: replace path with profile on machine! Needs to be done by user!
        path = "F:\Programming\Project\git\Code\Python\spherical_harmonics_generator\harmonics/"
        file = open(path + "spher_har_l_"+str(l)+".dat","wb")
        pickle.dump(l_arr,file)
        file.close()
    
    return arrays

                    
array = calculate_harmonics(l0,l1)

