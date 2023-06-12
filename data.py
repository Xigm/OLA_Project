# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 18:04:33 2023
@author: Xigm

"""

import numpy as np

prices = [10,25,40,55,70]

conv_rates_not_working = [0.7, 0.5, 0.3, 0.2, 0.1]

conv_rates_work_casual = [0.8, 0.6, 0.5, 0.4, 0.2]

conv_rates_work_hc =  [0.2, 0.4, 0.7, 0.8, 0.6]

CR = []

CR.append({i:j for i,j in zip(prices,conv_rates_not_working)})
CR.append({i:j for i,j in zip(prices,conv_rates_work_casual)})
CR.append({i:j for i,j in zip(prices,conv_rates_work_hc)})

data_points = np.arange(1,100)
bid_per_click = np.floor(100*np.log(data_points))

# clicks per bid
ClicksPB = []

ClicksPB.append({i:j for i,j in zip(10*bid_per_click,conv_rates_not_working)})
ClicksPB.append({i:j for i,j in zip(3*bid_per_click,conv_rates_work_casual)})
ClicksPB.append({i:j for i,j in zip(bid_per_click,conv_rates_work_hc)})

# cost per bid
CostPB = []

CostPB.append({i:j for i,j in zip(bid_per_click,conv_rates_not_working)})
CostPB.append({i:j for i,j in zip(bid_per_click,conv_rates_work_casual)})
CostPB.append({i:j for i,j in zip(bid_per_click,conv_rates_work_hc)})

def get_data():
    return CR, ClicksPB, CostPB