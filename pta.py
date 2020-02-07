# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 15:02:25 2020

@author: yashc
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2)
w0 = np.random.uniform(-0.25, 0.25)
w1 = np.random.uniform(-1, 1)
w2 = np.random.uniform(-1, 1)
W = np.array((w0, w1, w2)).reshape((1, 3))
print("The weights w0, w1, w2 picked are:")
print("w0 = {0}\nw1 = {1}\nw2 = {2}".format(w0, w1, w2))
S = np.empty((100, 2))

#for ele in np.nditer(S, op_flags = ['readwrite']):
#    ele[...] = np.random.uniform(-1, 1)

for row in range(np.shape(S)[0]):
    S[row] = np.random.uniform(-1, 1, 2)

bias = np.array((1))
S0 = np.empty((0, 2))
S1 = np.empty((0, 2))

for row in range(np.shape(S)[0]):
    if np.dot(np.append(bias, S[row]), W.T) < 0:
        S0 = np.append(S0, S[row].reshape((1, 2)), axis = 0)
    else:
        S1 = np.append(S1, S[row].reshape((1, 2)), axis = 0)

plt.scatter(*zip(*S0), marker = 'x', color = 'r', label = 'S0')
plt.scatter(*zip(*S1), marker = '.', color = 'b', label = 'S1')

x1 = np.linspace(-1.1, 1.6)
x2 = - (w1*x1 + w0) / w2
plt.ylim(-1.1, 1.1)
plt.plot(x1, x2, label = 'Classifier', color ='g')
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()