# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 15:02:25 2020

@author: yashc
"""

import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(100)


def ideal_w():
    global w0, w1, w2
    w0 = np.random.uniform(-0.25, 0.25)
    w1 = np.random.uniform(-1, 1)
    w2 = np.random.uniform(-1, 1)
    W = np.array((w0, w1, w2)).reshape((1, 3))
    print("The weights w0, w1, w2 picked are:")
    print("w0 = {0}\nw1 = {1}\nw2 = {2}\n".format(w0, w1, w2))
    return W


def set_S():
    S_100 = np.empty((100, 2))
    for row in range(np.shape(S_100)[0]):
        S_100[row] = np.random.uniform(-1, 1, 2)
    S_1000 = np.empty((1000, 2))
    for row in range(np.shape(S_1000)[0]):
        S_1000[row] = np.random.uniform(-1, 1, 2)
    return (S_100, S_1000)


def initial_w():
    global w0_prime, w1_prime, w2_prime
    w0_prime = np.random.uniform(-1, 1)
    w1_prime = np.random.uniform(-1, 1)
    w2_prime = np.random.uniform(-1, 1)
    print("The weights w0', w1', w2' picked are:")
    print("w0' = {0}\nw1' = {1}\nw2' = {2}\n".format(w0_prime, w1_prime, w2_prime))


def subset_S0_S1(S, W, n):
    print("\n-----------------------------------------------------------------------")
    print("For N =", n)
    global bias, S0, S1
    bias = np.array((1))
    S0 = np.empty((0, 2))
    S1 = np.empty((0, 2))

    for row in range(np.shape(S)[0]):
        if np.dot(np.append(bias, S[row]), W.T) < 0:
            S0 = np.append(S0, S[row].reshape((1, 2)), axis=0)
        else:
            S1 = np.append(S1, S[row].reshape((1, 2)), axis = 0)
    plt.scatter(*zip(*S0), marker = 'x', color = 'r', label = 'S0')
    plt.scatter(*zip(*S1), marker = '.', color = 'b', label = 'S1')

    x1 = np.linspace(-1.1, 1.6)
    x2 = - (W[0][1]*x1 + W[0][0]) / W[0][2]
    plt.ylim(-1.1, 1.1)
    plt.plot(x1, x2, label = 'Classifier', color ='g')
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.show()


# from (j)
def pta(S):
    eta_list = [1, 10, 0.1]

    for eta in eta_list:
        W_prime = np.array((w0_prime, w1_prime, w2_prime)).reshape((1, 3))
        print("-----------------------------------------------------------------------")
        print("For eta:", eta)
        epoch = 0
        no_misclassifications = 0
        misclassifications_list = []

        while True:
            no_misclassifications = 0
            epoch += 1
            for row in range(np.shape(S)[0]):
                if np.dot(np.append(bias, S[row]), W_prime.T) < 0:
                    if S[row] in S0:
                        continue
                    else:
                        no_misclassifications += 1
                        W_prime = W_prime + eta*np.append(bias, S[row])
                else:
                    if S[row] in S1:
                        continue
                    else:
                        no_misclassifications += 1
                        W_prime = W_prime - eta*np.append(bias, S[row])
            misclassifications_list.append(no_misclassifications)
            if no_misclassifications == 0:
                break
            else:
                continue
        print("Total number of epochs:", epoch, "\n")
        print("The final weights are :")
        print("w0 = {0}\nw1 = {1}\nw2 = {2}\n".format(W_prime[0][0], W_prime[0][1], W_prime[0][2]))
        print("Difference between these weights and the optimal weights are:")
        print("w0 difference: {0}\nw1 difference: {1}\nw2 difference: {2}\n".format(W_prime[0][0] - w0, W_prime[0][1] - w1, W_prime[0][2] - w2))

        plt.plot(range(epoch), misclassifications_list)
        plt.xlabel("Number of epochs")
        plt.ylabel("Number of misclassifications")
        plt.show()


if __name__ == "__main__":
    W = ideal_w()
    S_100, S_1000 = set_S()
    initial_w()
    subset_S0_S1(S_100, W, 100)
    pta(S_100)
    subset_S0_S1(S_1000, W, 1000)
    pta(S_1000)


