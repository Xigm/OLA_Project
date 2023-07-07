# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 18:26:14 2023
@author: Xigm

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from tqdm import tqdm 

def n(x):
    return (1 - np.exp(-5 * x))*100

def generate_observation(x,noise_std):
    return n(x) + np.random.normal(0,noise_std,size = n(x).shape)

n_obs = 50
bids = np.linspace(0, 1, 100)
x_obs = np.array([])
y_obs = np.array([])

noise_std = 5

for i in tqdm(range(n_obs)):
    
    new_x_obs = np.random.choice(bids,1)
    new_y_obs = generate_observation(new_x_obs, noise_std)
    
    x_obs = np.append(x_obs, new_x_obs)
    y_obs = np.append(y_obs, new_y_obs)
    
    X = np.atleast_2d(x_obs).T
    Y = y_obs.ravel()
    
    theta = 1
    l = 1
    kernel = C(theta, (1e-3,1e3)) * RBF(1,(1e-3,1e3))
    gp = GaussianProcessRegressor(kernel = kernel, alpha = 5, normalize_y=True, n_restarts_optimizer=10)
    
    gp.fit(X,Y)
    
    x_pred = np.atleast_2d(bids).T
    y_pred, sigma = gp.predict(x_pred, return_std = True)
    
    plt.figure()
    plt.plot(x_pred,n(x_pred))
    plt.plot(X.ravel(), Y, 'ro', label = u'Observed Clicks')
    plt.plot(x_pred, y_pred, 'b-', label = u'Predicted labels')
    plt.fill(np.concatenate([x_pred,x_pred[::-1]]),
             np.concatenate([y_pred - 1.96 * sigma, (y_pred + 1.96*sigma)[::-1]]),
             alpha = 0.5, fc = 'b', ec = 'None', label = "95% conf interval")
    plt.xlabel('$x$')
    plt.ylabel("$n(x)$")
    plt.show()
    
    