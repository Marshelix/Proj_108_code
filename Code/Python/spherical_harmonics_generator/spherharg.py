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
from datetime import datetime #for timings
import os
import gc
import psutil
import sys
mailpath = os.path.abspath(os.path.join(os.path.dirname(__file__),".."+"/py_mail"))
sys.path.append(mailpath)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),".."+"/Setup")))
from mailbot import email_bot
from Setup import setup
start_time = time.time()
print("Program started at "+str(datetime.now()))
#Setup
print("============= Load Settings =============")
settings = setup.load_settings()
print(settings)
print("============= Set up Email  =============")
#email
username = settings["Email"][4]
password = settings["Email"][3]
server = settings["Email"][2]

bot = email_bot(settings["Email"][0],settings["Email"][1],server,password,username,int(settings["Email"][5]))
bot.set_topic("Program Started")
bot.append_message("Program has commenced at " + str(datetime.now())) 

pid = os.getpid()
psu = psutil.Process(pid)
print("============= Find l0,l1   =============")
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
    #print("RAM usage before loop: "+str(psu.memory_info()[0]/2**30) + " Gb")
    bot.set_topic("Error Occured!")
    
    error_counter = 0   #increment on each nonfatal error or occurence, if > 3 abort
    for l in range(l0,l1):
        print("===============")
        print("l = " +str(l))
        l_start = time.time()
        #print("RAM usage at the start of calculation: "+str(psu.memory_info()[0]/2**30) + " Gb")
        l_arr = []
        #print("RAM usage after array reset: "+str(psu.memory_info()[0]/2**30) + " Gb")
        m_start = time.time()
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
        m_elapsed = time.time() - m_start
        print("Time elapsed on m-loop: "+str(m_elapsed)+" s")

        #note: replace path with profile on machine! Needs to be done by user!
        path = settings["Spherharg"][2]
        file = open(path + "spher_har_l_"+str(l)+".dat","wb")
        dump_start = time.time()
        pickle.dump(l_arr,file)
        dump_elapsed = time.time() - dump_start
        print("Time elapsed on dump: " + str(dump_elapsed) + " s")
        #print("RAM usage before closing the file: "+str(psu.memory_info()[0]/2**30) + " Gb")
        file.close()
        #print("RAM usage before collection: "+str(psu.memory_info()[0]/2**30) + " Gb")
        gc.collect()
        #print("RAM usage after collection: "+str(psu.memory_info()[0]/2**30) + " Gb")
        #print("File "+path + "spher_har_l"+str(l)+".dat"+" generated and saved.")
        l_end = time.time()
        l_loop_elapsed = l_end - l_start
        print("Time taken on l-loop for l = "+str(l)+": "+str(l_loop_elapsed)+" s")
        print("Percentages: ")
        print("m-loop: "+str(100*m_elapsed/l_loop_elapsed) + " %")
        print("dumping file: "+str(100*dump_elapsed/l_loop_elapsed)+" %")
        
        #send mail if too much harddrive is used up
        if (psutil.cpu_percent()) > 95:
            bot.append_message("["+str(datetime.now())+"]: "+"Too high cpu usage repeatedly at l =" + str(l))
            print("["+str(datetime.now())+"]: "+"Too high cpu usage repeatedly at l =" + str(l))
            error_counter = error_counter+1
        if (psu.memory_info()[0]/2**30)*100/16 > 95:
            #Ram keeps going up steadily, but slowly. Stop now.
            bot.append_message("["+str(datetime.now())+"]: "+"Too much ram space used at l =" + str(l))
            print("["+str(datetime.now())+"]: "+"Too much ram space used at l =" + str(l))
            error_counter = error_counter+1
        if psutil.disk_usage(path)[3] > 98:
            #no more disk space for next file: Stop calculation.
            bot.append_message("["+str(datetime.now())+"]: "+"Too much harddrive space used at l =" + str(l))
            print("["+str(datetime.now())+"]: "+"Too much harddrive space used at l =" + str(l))
            return 0
        if error_counter > 3:
            return 0
        
    return 0

                    
calculate_harmonics(l0,l1)
end_time = time.time()
elapsed = end_time - start_time
print("<===========================>")
print(str(elapsed) + " s = "+str(elapsed/60) +
      " min = "+str(elapsed/3600) +" h = "+
      str(elapsed/(3600*24)) +" days passed")
bot.set_topic("Program completed!")
bot.msg += "Current hard drive usage: "+str(psutil.disk_usage(settings["Spherharg"][2])[3]) +" %. \n"
bot.append_message("Calculation finished at "+"["+str(datetime.now())+"]")