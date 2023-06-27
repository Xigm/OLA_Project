# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 11:50:10 2023
@author: Xigm

"""

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from TS_learner import TS_Learner
from TS_learner_modded import TS_Learner_modded
from Greedy_learner import Greedy_learner
from Greedy_learner_modded import Greedy_learner_modded
from environment_pricing import environment_pricing
from tqdm import tqdm
from data import get_data

# Function to maximize
gains = lambda daily_clicks, conversion, margin, cumulative_daily_costs: daily_clicks * conversion * margin - cumulative_daily_costs

# Generate random conversion rates for user class C1 for prices 10, 15 and 20e for example
C1 = np.abs(np.random.randn(3))
C1 = C1/np.max(C1)/1.25
prices = [10,15,20]

C1, _, _ = get_data()
C1 = C1[1]
prices = list(C1.keys())

# Set the number of arms and the optimal conversion rate
n_arms = len(C1)
opt = max([i*j for i,j in zip(list(C1.values()),list(C1.keys()))])

# Set the number of rounds and experiments
T = 300
n_experiments = 500

# Initialize lists to store rewards for TS and Greedy learners
ts_rewards_p_exp = []
gr_rewards_p_exp = []
pulled_arms_ts = []
pulled_arms_gr = []

# Run experiments
for i in tqdm(range(n_experiments), desc="Experiments"):
    # Create the environment with pricing information
    # list(C1.values())
    env = environment_pricing(n_arms, list(C1.values()))
    
    # Initialize TS and Greedy learners
    ts_learner = TS_Learner_modded(n_arms,max(prices))
    gr_learner = Greedy_learner_modded(n_arms, max(prices))
    
    # Run rounds
    for t in range(T):
        # TS learner
        pulled_arm = ts_learner.pull_arm()
        pulled_arms_ts.append(pulled_arm)
        reward = env.next_round(pulled_arm)
        reward_com = reward*prices[pulled_arm]
        ts_learner.update(pulled_arm, reward_com)
        
        # Greedy learner
        pulled_arm = gr_learner.pull_arm()
        pulled_arms_gr.append(pulled_arm)
        reward = env.next_round(pulled_arm)
        reward_com = reward*prices[pulled_arm]
        gr_learner.update(pulled_arm, reward_com)
        
    # Store collected rewards for each learner
    ts_rewards_p_exp.append(ts_learner.collected_rewards)
    gr_rewards_p_exp.append(gr_learner.collected_rewards)
    
# Plot the cumulative regret
plt.figure()
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt-np.array(ts_rewards_p_exp),axis=0)),'r')
plt.plot(np.cumsum(np.mean(opt-np.array(gr_rewards_p_exp),axis=0)),'g')
plt.legend(["TS","Greedy"])

plt.figure()
plt.hist(pulled_arms_ts, alpha = 0.5)
plt.hist(pulled_arms_gr, alpha = 0.5)
plt.legend(["TS","Greedy"])

# Compute money earned
# VERY very doubtious
costs_of_manufacture = 10
daily_clicks = 100
index_best = np.argmax([sum(listi) for listi in np.array(gr_learner.rewards_per_arm,dtype = object)])
price_sell = prices[index_best]
conversion_rates_computed = [np.mean(listi) for listi in np.array(gr_learner.rewards_per_arm,dtype = object)]
margin = price_sell - costs_of_manufacture
cumulative_daily_costs = 100
money_earned = gains(daily_clicks,conversion_rates_computed[index_best],margin,cumulative_daily_costs)
print(money_earned)

