# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 14:39:17 2018

@author: Martin Sanner

Function file for a custom implementation of spherical harmonics.
This file covers:
    P_lsave(l,x):
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

'''

    So far, nothing in here works!!!!!!!!!!



'''
def P_lsave(l,m,x):
    '''
        Find the legendre polynomial of degree l for values of x
        cast data into dataframe, and if file doesnt exist, save to file
    '''
    filename = "leg_l_"+str(l)+"_m_"+str(m)+".dat"
    leg_pol_data = []
    cols = [str(m)]
    if type(x) == type([]):
        ind = np.arange(len(x))
    else:#
        ind = [0]
    df = pd.DataFrame(0,index = ind, columns = cols)
    print(datapath_leg+filename)
    if not os.path.exists(datapath_leg+filename) or os.path.getsize(datapath_leg+filename) == 0:
        #file doesnt yet exist, save to file, or exists but is empty.
        data = []
        for elem in x:
            leg_pol_data =  func_base.lpmn(m,l,elem)[0]
            data.append(leg_pol_data)
            print(leg_pol_data)
            print(leg_pol_data.shape)
            print(type(leg_pol_data))
        print(data)
        data = np.array(data)
        print(type(data))
        print(len(data))
        df = pd.DataFrame(data,index = ind, columns = cols)
        file = open(datapath_leg+filename,"wb")
        pickle.dump(df,file)
        file.close()
    else:
         if os.path.getsize(datapath_leg+filename) > 0:
            print("Loading from file.")
            file = open(datapath_leg+filename,"rb")
            df = pickle.load(file)
            file.close()
    return df
'''
Spherical Harmonics
'''
def spher_harm_custom(l,m,theta,phi):
    '''
    returns a dataframe of the spherical harmonics ylm using the builtin method for plm
    '''

    factor_1 = np.sqrt(((2*l+1)/4*np.pi)*(misc.factorial(l-m)/misc.factorial(m+l)))
  
   # vec_lpmn = np.vectorize(func_base.lpmn,excluded = ["m","l"])
    factor_2 = []
    #print("Theta type = "+str(type(theta)))
    for t in theta[0]:
        #print("t = "+str(t))
        #print("t-type = "+str(type(t)))
        #print(np.cos(t))
        ylm = func_base.lpmn(m,l,np.cos(t))[0]
        print("YLM type: "+str(type(ylm)))
        print("YLM len: "+str(len(ylm)))
        print(ylm)
        print(ylm[0])
        print(ylm[1])
        factor_2.append(ylm)
    fac = []
    for elem in factor_2:
        for e in elem:
            fac.append(e)
    print(fac)
    #factor_2 = vec_lpmn(m,l,np.cos(theta))[0]  #associated legendre polynomial for cos(theta)
    exponents = [1j*m*p for p in phi]   #Apparently can't multiply array of real values by j -> way around this
    factor_3 = np.exp(exponents)  #j used for imaginary units
    #print(factor_1)
    print("Factor_1 type: "+str(type(factor_1)))
    #print(factor_2)
    factor_2 = np.array(factor_2)
    print("Factor_2 type: "+str(type(factor_2)))
    print("Factor_2 shape: "+str(factor_2.shape))
    #print(factor_3)
    print("Factor_3 type: "+str(type(factor_3)))
    factor11 = factor_1*np.array(factor_2)
    print(factor11)
    print("Factor11  type: "+str(type(factor11)))
    print("Factor11 shape: "+str(factor11.shape))
    
    return factor11*factor_3
'''
N = int(2*np.pi*10)
theta = np.linspace(0,2*np.pi,N)
phi = np.linspace(0,np.pi,N)
theta,phi = np.meshgrid(theta,phi)
ylm = spher_harm_custom(2,1,theta,phi)
print(type(ylm))
print(ylm.shape)
time_end = time.time()
time_elapsed = time_end - start_time
print("Time elapsed = " + str(repr(time_elapsed)))
'''
'''
#testing spherical harmonics

l_max = 10
l = np.linspace(0,l_max,11)
m = np.linspace(0,l_max,11)
print(theta)
print(len(theta))
print(phi)
print(len(phi))
theta,phi = np.meshgrid(theta,phi)
ylm = func_base.sph_harm(l_max,l_max,theta,phi)
print(ylm)
print(type(ylm))
print(ylm.shape)
'''