# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 18:56:53 2023
@author: Xigm

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from tqdm import tqdm 

def n(x):
    # data distribution to fit to
    return (1 - np.exp(-5 * x))*100

def generate_observation(x,noise_std):
    return n(x) + np.random.normal(0,noise_std,size = x.shape)

noise_std = 10
alpha_values = np.logspace(-4,1,25)
db_alpha =10*np.log10(alpha_values)
bids = np.linspace(0, 1, 100)
x_obs = bids
y_obs = generate_observation(bids, noise_std)

X = np.atleast_2d(x_obs).T
Y = y_obs.ravel()
    
y_val = generate_observation(bids, noise_std)
Y_val = y_val.ravel()

val_scores = []
scores = []
for alpha in tqdm(alpha_values):

    theta = 1
    l = 1
    kernel = C(theta, (1e-3,1e3)) * RBF(1,(1e-3,1e3))
    gp = GaussianProcessRegressor(kernel = kernel, alpha = alpha, normalize_y=True, n_restarts_optimizer=10)
    
    gp.fit(X,Y)
    
    scores.append(gp.score(X,Y))
    val_scores.append(gp.score(X,Y_val))
    
    x_pred = np.atleast_2d(bids).T
    y_pred, sigma = gp.predict(x_pred, return_std = True)
    
    # plt.figure()
    # plt.plot(x_pred,n(x_pred))
    # plt.plot(X.ravel(), Y, 'ro', label = u'Observed Clicks')
    # plt.plot(x_pred, y_pred, 'b-', label = u'Predicted labels')
    # plt.fill(np.concatenate([x_pred,x_pred[::-1]]),
    #          np.concatenate([y_pred - 1.96 * sigma, (y_pred + 1.96*sigma)[::-1]]),
    #          alpha = 0.5, fc = 'b', ec = 'None', label = "95% conf interval")
    # plt.xlabel('$x$')
    # plt.ylabel("$n(x)$")
    # plt.show()
    
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

plt.figure()
plt.plot(db_alpha, scores,'r-', label = u'Scores Train')
plt.plot(db_alpha, scores,'b-', label = u'Scores Val')

plt.xlabel('$ Alpha $ values [dB]')
plt.ylabel("$Score$")
plt.legend()
plt.show()
