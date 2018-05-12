# -*- coding: utf-8 -*-
"""
Created on Wed May  9 17:21:20 2018

@author: marti
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


sigma = 100
lambd  = 4
phi = np.linspace(-3*sigma,3*sigma,1000)
fig = plt.figure()
ax = fig.gca(projection = '3d')
X,Y = np.meshgrid(phi,phi)
V = -lambd/4 * (X*Y - sigma**2)**2
ax.plot_surface(X,Y,V)

ax.set_title("Wine bottle Potential")