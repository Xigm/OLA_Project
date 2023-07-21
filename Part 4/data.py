# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 18:04:33 2023
@author: Xigm

"""

import numpy as np
import matplotlib.pyplot as plt

prices = [10,25,40,55,70]

conv_rates_not_working = [0.7, 0.5, 0.3, 0.2, 0.1]

conv_rates_work_casual = [0.8, 0.6, 0.5, 0.4, 0.2]

conv_rates_work_hc =  [0.2, 0.4, 0.7, 0.8, 0.6]

CR = []

CR.append({i:j for i,j in zip(prices,conv_rates_not_working)})
CR.append({i:j for i,j in zip(prices,conv_rates_work_casual)})
CR.append({i:j for i,j in zip(prices,conv_rates_work_hc)})

L = 100
data_points = np.arange(1,L)/L
bid_per_click_log = 25*np.log(data_points*L)
bid_per_click_exp = 100* (1 - np.exp(-4*data_points+3*data_points**3))

# clicks per bid
ClicksPB = []

th = 20
ClicksPB.append({i:j for i,j in zip(data_points,np.r_[np.zeros(th),5*bid_per_click_exp[:L-th]])})
th = 10
ClicksPB.append({i:j for i,j in zip(data_points,np.r_[np.zeros(th),3*bid_per_click_exp[:L-th]])})
th = 50
ClicksPB.append({i:j for i,j in zip(data_points,np.r_[np.zeros(th),1*bid_per_click_exp[:L-th]])})

# cost per bid
CostPB = []

th = 20
CostPB.append({i:j for i,j in zip(data_points,np.r_[np.zeros(th),1*bid_per_click_log[:L-th]])})
th = 10
CostPB.append({i:j for i,j in zip(data_points,np.r_[np.zeros(th),5*bid_per_click_log[:L-th]])})
th = 50
CostPB.append({i:j for i,j in zip(data_points,np.r_[np.zeros(th),3*bid_per_click_log[:L-th]])})

def get_data():
    return CR, ClicksPB, CostPB