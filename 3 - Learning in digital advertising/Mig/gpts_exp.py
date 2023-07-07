# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 19:56:26 2023
@author: Xigm

"""

import numpy as np
import matplotlib.pyplot as plt
from bidding_env import *
from gts_learner import *
from gpts_learner import *
from tqdm import tqdm

import warnings
from sklearn.exceptions import ConvergenceWarning

# Ignore scikit-learn warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)


n_arms = 20
min_bid = 0.0
max_bid = 1.0
bids = np.linspace(min_bid, max_bid, n_arms)
sigma = 10

T = 60
n_experiments = 30
gts_rewards_per_experiment = []
gpts_rewards_per_experiment = []
gpts2_rewards_per_experiment = []

scores = []
scores2 = []
for i in tqdm(range(n_experiments), desc="Experiments"):
    env = bidding_env(bids=bids, sigma=sigma)
    Gts_learner = gts_learner(n_arms=n_arms)
    Gpts_learner = gpts_learner(n_arms=n_arms, arms=bids, theta = 1, alpha = 0.01)
    Gpts_learner2 = gpts_learner(n_arms=n_arms, arms=bids, theta = 1, alpha = 100)
    
    for t in tqdm(range(T), desc = "Inner loop"):
        # Gaussian Thompson Sampling
        pulled_arm = Gts_learner.pull_arm()
        reward = env.round(pulled_arm)
        Gts_learner.update(pulled_arm, reward)

        # GP Thompson Sampling
        pulled_arm = Gpts_learner.pull_arm()
        reward = env.round(pulled_arm)
        Gpts_learner.update(pulled_arm, reward)
        
        # GP Thompson Sampling
        pulled_arm = Gpts_learner2.pull_arm()
        reward = env.round(pulled_arm)
        Gpts_learner2.update(pulled_arm, reward)

    scores.append(Gpts_learner.score())
    scores2.append(Gpts_learner2.score())
    gts_rewards_per_experiment.append(Gts_learner.collected_rewards)
    gpts_rewards_per_experiment.append(Gpts_learner.collected_rewards)
    gpts2_rewards_per_experiment.append(Gpts_learner2.collected_rewards)

opt = np.max(env.means)
plt.figure(0)
plt.ylabel('Regret')
plt.xlabel('t')
plt.plot(np.cumsum(np.mean(opt - gts_rewards_per_experiment, axis=0)), 'r')
plt.plot(np.cumsum(np.mean(opt - gpts_rewards_per_experiment, axis=0)), 'g')
plt.plot(np.cumsum(np.mean(opt - gpts2_rewards_per_experiment, axis=0)), 'b')
plt.legend(['GTS', 'GPTS alpha nice', 'GPTS his alpha'])
plt.show()