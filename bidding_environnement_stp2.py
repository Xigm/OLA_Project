# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 03:47:43 2023

@author: saade
"""

import numpy as np 

def n(x):
    return 100 * (1-np.exp(-4*x+3*x**3))

def C(x):
    return 100 * (1-np.exp(-2*x+x**3))

def gain(x, marg, rate):
    return marg*rate*n(x)-C(x)



class BiddingEnvironment() : 
    def __init__(self, bids, sigma, margin, Conv_rate):
        self.bids = bids
        self.means = gain(bids,margin, Conv_rate)
        self.sigmas = np.ones(len(bids)) * sigma
        
    def round(self, pulled_arm):
        return np.random.normal(self.means[pulled_arm], self.sigmas[pulled_arm])
    
        