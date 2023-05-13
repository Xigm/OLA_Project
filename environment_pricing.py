# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 11:42:27 2023
@author: Xigm

"""
import numpy as np

class environment_pricing():
    def __init__(self,n_arms,prob):
        self.n_arms = n_arms
        self.prob = prob
        
    def next_round(self,pulled_arm):
        # MIG: we should take into account the price of the products
        return np.random.binomial(1,self.prob[pulled_arm])
            