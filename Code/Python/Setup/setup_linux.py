# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 10:15:07 2018

@author: Martin Sanner

Setup file: Creates  setup.dat, a binary dataframe of setup information. On start tries to open the file using pickle
"""

import pandas as pd
import pickle
import numpy as np
import os


def load_settings():
    '''
    Load a settings file. If it doesnt exist, create one and set it up
    
    '''
    filename = "settings2.dat"
    index = range(0,7)
    col = ["Email","Spherharg","NN"]
    settings = pd.DataFrame(0,index = index,columns=col )
    if not os.path.exists('./'+filename):
        print("Create settings")
        return create_settings()
    else:
        if os.path.getsize('./'+filename) > 0:
            print("Loading Settings from file.")
            file = open('./'+filename,"rb")
            settings = pickle.load(file)
            file.close()
    
    return settings

def create_settings():
    filename = "settings.dat"
    index = range(0,7)
    col = ["Email","Spherharg","NN"]
    settings = pd.DataFrame(0,index = index,columns=col )
    

    if not os.path.exists('./'+filename):
        print("Creating new file")
        file = open('./'+filename,"wb")
        file.close()
    else:
        if os.path.getsize('./'+filename) > 0:
            print("Loading from file.")
            file = open('./'+filename,"rb")
            settings = pickle.load(file)
            file.close()
    
    print("To use, enter commands via the console below.")
    print("Commands can be set as 'set descriptor index data', separated by spaces")
    print("Possible commands: set, read, quit, save" )
    print("Possible descriptors: "+str(col) +"and 'all'")
    print("Possible indices: 0-6")
    com = input("Enter command(quit to stop):")

    while com != "quit":
        values = com.split(" ")
        if com == "save":
            print("Saving.")
            file = open('./'+filename,"wb")
            pickle.dump(settings,file)
            file.close()
            print("File saved")
        if len(values) == 4:
            command = values[0]
            descriptor = values[1]
            index = int(values[2])
            val = values[3]
        
            if descriptor in col or descriptor == "all" and index in np.linspace(0,6,7) and (command == "set" or command == "read" or command == "save"):
               
                if command == "set":
                    desc = descriptor
                    idx = index
                    settings[desc][idx] = val
                    print(settings)
                elif command == "read":
                    desc = descriptor
                    if desc == "all":
                        print(settings)
                    else:
                        idx = index
                        print(settings[desc][idx])
                elif command == "save":
                    print("Saving.")
                    file = open('./'+filename,"wb")
                    pickle.dump(settings,file)
                    file.close()
                    print("File saved")
        elif len(values) == 1:
            if values[0] == "quit":
                continue
            
        com = input("Enter command(quit to stop):")
    return settings
create_settings()