import numpy as np
import matplotlib.pyplot as plt
from TS_learner_modded import TS_Learner_modded
from environment_pricing import environment_pricing
from tqdm import tqdm
from data import get_data
from BiddingEnvironment_Step3 import *
from GPTS_Learner_Step3 import *
from GPUCB_Learner_Step3 import *
from sklearn.exceptions import ConvergenceWarning
import warnings

# ignore scikit-learn warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Function to maximize
gains = lambda daily_clicks, conversion, margin, cumulative_daily_costs: daily_clicks * conversion * margin - cumulative_daily_costs

C1, clicks, cost = get_data()
C1 = C1[1]
clicks = clicks[1]
cost = cost[1]
bids = list(clicks.keys())
prices = list(C1.keys())
convertion_rate = list(C1.values())

# Set the number of arms and the optimal conversion rate and profit
n_arms_price = len(C1)
n_arms_adv = len(clicks)
opt_price = max([i*j for i,j in zip(list(C1.values()),list(C1.keys()))])
opt_profit = max([opt_price * i - z for i,j,z in zip(list(clicks.values()), list(clicks.keys()), list(cost.values()))])

# Set the number of rounds and experiments
# Mig: From around day 150 behaviour is linear
T = 365
n_experiments = 5
sigma = 10

# Initialize lists to store rewards for GP-TS and GP-UCB
gpts_rewards_per_experiment = []
gpucb_rewards_per_experiment = []

for i in tqdm(range(n_experiments), desc="Experiments"):
    # Create price and adv environment
    env_price = environment_pricing(n_arms_price, list(C1.values()))
    env_bid = BiddingEnvironment(bids=bids, sigma=sigma, clicks=clicks, cost=cost)

    # Initialize Learners
    ts_learner_price = TS_Learner_modded(n_arms_price,max(prices))
    #gr_learner = Greedy_learner_modded(n_arms, max(prices))
    gpts_learner = GPTS_Learner(n_arms=n_arms_adv, arms=bids)
    gpucb_learner = GPUCB_Learner(n_arms=n_arms_adv, arms=bids)

    # Run rounds
    for t in tqdm(range(T), desc='Day'):

        # Price TS Round (We decide a price)
        pulled_arm_price = ts_learner_price.pull_arm()
        reward = env_price.next_round(pulled_arm_price)
        reward_com = reward*prices[pulled_arm_price]
        ts_learner_price.update(pulled_arm_price, reward_com)

        # GPTS Round
        pulled_arm_adv = gpts_learner.pull_arm()
        reward_adv = env_bid.round(pulled_arm_adv, conv_rate=convertion_rate[pulled_arm_price], price=prices[pulled_arm_price])
        gpts_learner.update(pulled_arm_adv, reward_adv)

        # GPUCB Round
        pulled_arm_adv = gpucb_learner.pull_arm()
        reward_adv = env_bid.round(pulled_arm_adv, conv_rate=convertion_rate[pulled_arm_price], price=prices[pulled_arm_price])
        gpucb_learner.update(pulled_arm_adv, reward_adv)

    # Store collected rewards for each learner
    gpts_rewards_per_experiment.append(gpts_learner.collected_rewards)
    gpucb_rewards_per_experiment.append(gpucb_learner.collected_rewards)


print('Optimal Profit: ',opt_profit*T)
print('GP-TS Profit: ',max(np.cumsum(np.mean(np.array(gpts_rewards_per_experiment),axis=0))))
print('GP-UCB Profit: ',max(np.cumsum(np.mean(np.array(gpucb_rewards_per_experiment),axis=0))))

# Plot the cumulative regret
plt.figure("Cumulative Regret")
plt.xlabel("t")
plt.ylabel("Cumulative Regret")
plt.plot(np.cumsum(np.mean(opt_profit - np.array(gpts_rewards_per_experiment),axis=0)),'r')
plt.plot(np.cumsum(np.mean(opt_profit - np.array(gpucb_rewards_per_experiment),axis=0)),'g')
plt.legend(["GP-TS","GP-UCB"])

# plot the cumulative profit
plt.figure('Cumulative Reward')
plt.plot(np.cumsum(np.mean(np.array(gpts_rewards_per_experiment),axis=0)),'r')
plt.plot(np.cumsum(np.mean(np.array(gpucb_rewards_per_experiment),axis=0)),'g')
plt.plot(np.cumsum(np.full(T, opt_profit)), 'b')
plt.xlabel("t")
plt.ylabel("Profit")
plt.legend(["GP-TS","GP-UCB","Optimal"])

# plot the Instantaneous Regret
plt.figure("Instantaneous Regret")
plt.xlabel("t")
plt.ylabel("Instantaneous Regret")
plt.plot(np.mean(opt_profit - np.array(gpts_rewards_per_experiment),axis=0),'r')
plt.plot(np.mean(opt_profit - np.array(gpucb_rewards_per_experiment),axis=0),'g')
plt.legend(["GP-TS","GP-UCB"])

# plot the Instantaneous profit
plt.figure('Instantaneous Reward')
plt.plot(np.mean(np.array(gpts_rewards_per_experiment),axis=0),'r')
plt.plot(np.mean(np.array(gpucb_rewards_per_experiment),axis=0),'g')
plt.plot(np.full(T, opt_profit), 'b')
plt.xlabel("t")
plt.ylabel("Profit")
plt.legend(["GP-TS","GP-UCB","Optimal"])
plt.show()



