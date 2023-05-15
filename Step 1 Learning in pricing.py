# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 11:59:50 2023
@author: Xigm
"""

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from TS_learner import TS_Learner
from Greedy_learner import Greedy_learner
from environment_pricing import environment_pricing
from tqdm import tqdm

# Function to maximize
gains = lambda daily_clicks, conversion, margin, cumulative_daily_costs: daily_clicks * conversion * margin - cumulative_daily_costs

# Generate random conversion rates for user class C1 for prices 10, 15 and 20e for example
C1 = np.abs(np.random.randn(3))
C1 = C1/np.max(C1)/1.25
prices = [10,15,20]

# Set the number of arms and the optimal conversion rate
n_arms = len(C1)
opt = max(C1)

# Set the number of rounds and experiments
T = 300
n_experiments = 150

# Initialize lists to store rewards for TS and Greedy learners
ts_rewards_p_exp = []
gr_rewards_p_exp = []

# Run experiments
for i in tqdm(range(n_experiments), desc="Experiments"):
    # Create the environment with pricing information
    env = environment_pricing(n_arms, C1)
    
    # Initialize TS and Greedy learners
    ts_learner = TS_Learner(n_arms)
    gr_learner = Greedy_learner(n_arms)
    
    # Run rounds
    for t in range(T):
        # TS learner
        pulled_arm = ts_learner.pull_arm()
        reward = env.next_round(pulled_arm)
        ts_learner.update(pulled_arm, reward)
        
        # Greedy learner
        pulled_arm = gr_learner.pull_arm()
        reward = env.next_round(pulled_arm)
        gr_learner.update(pulled_arm, reward)
        
    # Store collected rewards for each learner
    ts_rewards_p_exp.append(ts_learner.collected_rewards)
    gr_rewards_p_exp.append(gr_learner.collected_rewards)
    
# Plot the cumulative regret
plt.figure()
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt-ts_rewards_p_exp,axis=0)),'r')
plt.plot(np.cumsum(np.mean(opt-gr_rewards_p_exp,axis=0)),'g')
plt.legend(["TS","Greedy"])


# Compute money earned
# VERY very doubtious
costs_of_manufacture = 5
daily_clicks = 100
index_best = np.argmax([sum(listi) for listi in np.array(gr_learner.rewards_per_arm,dtype = object)])
price_sell = prices[index_best]
conversion_rates_computed = [np.mean(listi) for listi in np.array(gr_learner.rewards_per_arm,dtype = object)]
margin = price_sell - costs_of_manufacture
cumulative_daily_costs = 100
money_earned = gains(daily_clicks,conversion_rates_computed[index_best],margin,cumulative_daily_costs)
print(money_earned)

