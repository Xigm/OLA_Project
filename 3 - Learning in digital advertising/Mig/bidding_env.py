# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 19:35:58 2023
@author: Xigm

"""
import numpy as np



def fun(x):
    return 100 * (1 - np.exp(-4*x+3*x**3))


class bidding_env():
    def __init__(self,bids,sigma):
        self.bids = bids
        self.means = fun(bids)
        self.sigma = np.ones(len(bids))*sigma
        
    def round(self, pulled_arm):
        return np.random.normal(self.means[pulled_arm],self.sigma[pulled_arm])
    
        
        