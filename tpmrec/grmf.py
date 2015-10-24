#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Gaussian Ranking by Matrix Factorization
Harald Steck Recsys 2015


You can run following codes.
grmf = GRMF(R, inds, K, alpha, lam)
grmf.train(epochs=100)
grmf.predict()
"""

import numpy as np
import random


class GRMF():
    """
    GRMF()
    """

    def __init__(self, R, inds, K, alpha, lam, eta):
        """ init paramaters """
        self.R = R  # true value
        self.inds = inds  # index of true value
        self.K = K  # dimention of latent factor
        self.alpha = alpha  # training rate of SGD
        self.lam = lam  # regularized parameter of matrix factorization
        #N = len(R)  # number of user
        self.eta = eta
        M = len(R[0])  # number of item
        self.N = len(R)
        self.M = len(R[0])
        self.p = np.random.random([M, K])  # latent factor of item for output layer
        self.q = np.random.random([M, K])  # latent factor of item for input layer

    def train(self, epochs=10):
        for epoch in range(epochs):
            print "epoch=", epoch
            print "R_ = ", self.predict()
            R_ = self.predict()
            for u in range(self.N):
                A = 0
                e = np.zeros(self.K)
                v = np.sum(self.q[self.inds[u], :], axis=0) / np.sqrt(len(self.inds[u]))
                for i in range(self.M):
                    pv = np.dot(self.p[i, :].T, v)
                    #print self.p[i, :].T
                    #print v
                    #print pv
                    if i in self.inds[u]:
                        sigma = 1.0 / (1.0 + np.exp(-pv))
                        err = -sigma*(1 - sigma)
                        #err = -pv*(1 - pv)
                        lam = self.lam
                        A = A + self.lam
                    else:
                        if pv < 0:
                            pv = 0
                        err = -pv
                        lam = 0
                    e = e + err*self.p[i, :]/np.sqrt(len(self.inds[u]))
                    self.p[i, :] = self.p[i, :] + self.eta*(err*v - lam*self.p[i, :])
                for i in self.inds[u]:
                    self.q[i, :] = self.q[i, :] + self.eta*(e - A*self.q[i, :])
            #print self.p
            #print self.q
            print np.sqrt(np.sum((R_ - self.predict())**2) / (self.N*self.M))

    def predict(self):
        R_ = np.zeros([self.N, self.M])
        for u in range(self.N):
            v = np.sum(self.q[self.inds[u], :], axis=0) / np.sqrt(len(self.inds[u]))
            R_[u, :] = np.dot(v, self.q.T)
        return R_

if __name__ == "__main__":
    R = np.array([[1, 1, 1, 1],
                 [1, 0, 1, 0],
                 [0, 1, 0, 1],
                 [0, 1, 1, 1]])
    inds = [[0, 1], [0, 2, 3], [1, 3], [2, 3]]
    K = 2
    alpha = 0.01
    lam = 0.01
    eta = 0.3

    grmf = GRMF(R, inds, K, alpha, lam, eta)
    grmf.train(epochs=100)
    print grmf.predict()
