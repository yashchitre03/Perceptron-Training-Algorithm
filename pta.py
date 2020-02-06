# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 15:02:25 2020

@author: yashc
"""

import numpy as np


w0 = np.random.uniform(-0.25, 0.25)
w1 = np.random.uniform(-1, 1)
w2 = np.random.uniform(-1, 1)
W = np.array((w0, w1, w2))
print("The weights w0, w1, w2 picked are:")
print("w0 = {0}\nw1 = {1}\nw2 = {2}".format(w0, w1, w2))
S = np.empty((100, 2))

#for ele in np.nditer(S, op_flags = ['readwrite']):
#    ele[...] = np.random.uniform(-1, 1)

for row in range(np.shape(S)[0]):
    S[row] = np.random.uniform(-1, 1, 2)

