# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 04:28:55 2023

@author: saade
"""

import numpy as np
import matplotlib.pyplot as plt 
from BiddingEnvironment_Step2 import *
from GPTS_Learner_Step2 import *
from GPUCB_Learner_Step2 import *
from sklearn.exceptions import ConvergenceWarning
import warnings


# ignore scikit-learn warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

n_arms = 100
min_bid = 0
max_bid = 1

bids = np.linspace(min_bid,max_bid, n_arms)
sigma = 10

T = 365
n_experiments = 500

gpts_rewards_per_experiment = []
gpucb_rewards_per_experiment= []


#just an arbitrary value since we need to get it from a given curve 
M = 50  
cr = 0.6


for e in range(0, n_experiments):
    env = BiddingEnvironment(bids=bids, sigma=sigma, margin=M, Conv_rate=cr)
    gpts_learner = GPTS_Learner(n_arms=n_arms, arms=bids)
    gpucb_learner = GPUCB_Learner(n_arms=n_arms_adv, arms=bids)
    
    for t in range(0,T):
         # GPTS Round
        pulled_arm = gpts_learner.pull_arm()
        reward = env.round(pulled_arm)
        gpts_learner.update(pulled_arm, reward)


        # GPUCB Round
        pulled_arm = gpucb_learner.pull_arm()
        reward = env.round(pulled_arm)
        gpucb_learner.update(pulled_arm_adv, reward_adv)
    
    gpts_rewards_per_experiment.append(gpts_learner.collected_rewards)
    gpucb_rewards_per_experiment.append(gpucb_learner.collected_rewards)
    
    
print('GP-TS Profit: ',max(np.cumsum(np.mean(np.array(gpts_rewards_per_experiment),axis=0))))
print('GP-UCB Profit: ',max(np.cumsum(np.mean(np.array(gpucb_rewards_per_experiment),axis=0))))

opt = np.max(env.means)


# Plot the cumulative regret
plt.figure("Cumulative Regret")
plt.xlabel("t")
plt.ylabel("Cumulative Regret")
plt.plot(np.cumsum(np.mean(opt - np.array(gpts_rewards_per_experiment),axis=0)),'r')
plt.plot(np.cumsum(np.mean(opt - np.array(gpucb_rewards_per_experiment),axis=0)),'g')
plt.legend(["GP-TS","GP-UCB"])


# plot the cumulative profit
plt.figure('Cumulative Reward')
plt.plot(np.cumsum(np.mean(np.array(gpts_rewards_per_experiment),axis=0)),'r')
plt.plot(np.cumsum(np.mean(np.array(gpucb_rewards_per_experiment),axis=0)),'g')
plt.plot(np.cumsum(np.full(T, opt)), 'b')
plt.xlabel("t")
plt.ylabel("Profit")
plt.legend(["GP-TS","GP-UCB","Optimal"])

# plot the Instantaneous Regret
plt.figure("Instantaneous Regret")
plt.xlabel("t")
plt.ylabel("Instantaneous Regret")
plt.plot(np.mean(opt - np.array(gpts_rewards_per_experiment),axis=0),'r')
plt.plot(np.mean(opt - np.array(gpucb_rewards_per_experiment),axis=0),'g')
plt.legend(["GP-TS","GP-UCB"])

# plot the Instantaneous profit
plt.figure('Instantaneous Reward')
plt.plot(np.mean(np.array(gpts_rewards_per_experiment),axis=0),'r')
plt.plot(np.mean(np.array(gpucb_rewards_per_experiment),axis=0),'g')
plt.plot(np.full(T, opt), 'b')
plt.xlabel("t")
plt.ylabel("Profit")
plt.legend(["GP-TS","GP-UCB","Optimal"])
plt.show()
