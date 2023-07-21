import numpy as np
import matplotlib.pyplot as plt


from TS_learner_modded import TS_Learner_modded
#from Greedy_learner import Greedy_learner
#from Greedy_learner_modded import Greedy_learner_modded
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

C, clicks, cost = get_data()
C1 = C[0]
C2= C[1]
C3= C[2]

clicks1 = clicks[0]
cost1 = cost[1]

clicks2 = clicks[1]
cost2 = cost[1]

clicks3 = clicks[2]
cost3 = cost[2]



bids1 = list(clicks1.keys())
prices1 = list(C1.keys())
convertion_rate1 = list(C1.values())

bids2 = list(clicks2.keys())
prices2 = list(C2.keys())
convertion_rate2 = list(C2.values())

bids3 = list(clicks3.keys())
prices3 = list(C3.keys())
convertion_rate3 = list(C3.values())




# Set the number of arms and the optimal conversion rate and profit
n_arms_price1 = len(C1)
n_arms_adv1 = len(clicks1)
opt_price1 = max([i*j for i,j in zip(list(C1.values()),list(C1.keys()))])
opt_profit1 = max([opt_price1 * i - z for i,j,z in zip(list(clicks1.values()), list(clicks1.keys()), list(cost1.values()))])

n_arms_price2 = len(C2)
n_arms_adv2 = len(clicks2)
opt_price2 = max([i*j for i,j in zip(list(C2.values()),list(C1.keys()))])
opt_profit2 = max([opt_price2 * i - z for i,j,z in zip(list(clicks2.values()), list(clicks2.keys()), list(cost2.values()))])

n_arms_price3 = len(C3)
n_arms_adv3 = len(clicks3)
opt_price3 = max([i*j for i,j in zip(list(C3.values()),list(C3.keys()))])
opt_profit3 = max([opt_price3 * i - z for i,j,z in zip(list(clicks3.values()), list(clicks3.keys()), list(cost3.values()))])



# Set the number of rounds and experiments
# Mig: From around day 150 behaviour is linear
T = 450
n_experiments = 3
sigma1 = 10
sigma2 = 10
sigma3 = 10

# Initialize lists to store rewards for GP-TS and GP-UCB
gpts_rewards_per_experiment1 = []
gpucb_rewards_per_experiment1 = []

gpts_rewards_per_experiment2 = []
gpucb_rewards_per_experiment2 = []

gpts_rewards_per_experiment3 = []
gpucb_rewards_per_experiment3 = []


for i in tqdm(range(n_experiments), desc="Experiments"):
    # Create price and adv environment
    env_price1 = environment_pricing(n_arms_price1, list(C1.values()))
    env_bid1 = BiddingEnvironment(bids=bids1, sigma=sigma1, clicks=clicks1, cost=cost1)
    
    env_price2 = environment_pricing(n_arms_price2, list(C2.values()))
    env_bid2 = BiddingEnvironment(bids=bids2, sigma=sigma2, clicks=clicks2, cost=cost2)
    
    env_price3 = environment_pricing(n_arms_price3, list(C3.values()))
    env_bid3 = BiddingEnvironment(bids=bids3, sigma=sigma3, clicks=clicks3, cost=cost3)

    # Initialize Learners
    ts_learner_price1 = TS_Learner_modded(n_arms_price1,max(prices1))
    gpts_learner1 = GPTS_Learner(n_arms=n_arms_adv1, arms=bids1)
    gpucb_learner1 = GPUCB_Learner(n_arms=n_arms_adv1, arms=bids1)
    
    ts_learner_price2 = TS_Learner_modded(n_arms_price2,max(prices2))
    gpts_learner2 = GPTS_Learner(n_arms=n_arms_adv2, arms=bids2)
    gpucb_learner2 = GPUCB_Learner(n_arms=n_arms_adv2, arms=bids2)


    


    # Run rounds
    for t in tqdm(range(T), desc='Day'):

        select_class= np.random.randint(0,3)
        

        if select_class==0: 
            # Price TS Round (We decide a price)
            pulled_arm_price1 = ts_learner_price1.pull_arm()
            reward1 = env_price1.next_round(pulled_arm_price1)
            reward_com1 = reward1*prices1[pulled_arm_price1]
            ts_learner_price1.update(pulled_arm_price1, reward_com1)
            
            # GPTS Round
            pulled_arm_adv1 = gpts_learner1.pull_arm()
            reward_adv1 = env_bid1.round(pulled_arm_adv1, conv_rate=convertion_rate1[pulled_arm_price1], price=prices1[pulled_arm_price1])
            gpts_learner1.update(pulled_arm_adv1, reward_adv1)
            
            # GPUCB Round
            pulled_arm_adv1 = gpucb_learner1.pull_arm()
            reward_adv1 = env_bid1.round(pulled_arm_adv1, conv_rate=convertion_rate1[pulled_arm_price1], price=prices1[pulled_arm_price1])
            gpucb_learner1.update(pulled_arm_adv1, reward_adv1)
            
            # Store collected rewards for each learner
            gpts_rewards_per_experiment1.append(gpts_learner1.collected_rewards)
            gpucb_rewards_per_experiment1.append(gpucb_learner1.collected_rewards)
        
        
        elif select_class==1:
                 
            # Price TS Round (We decide a price)
            pulled_arm_price2 = ts_learner_price2.pull_arm()
            reward2 = env_price2.next_round(pulled_arm_price2)
            reward_com2 = reward2*prices2[pulled_arm_price2]
            ts_learner_price2.update(pulled_arm_price2, reward_com2)
            
            # GPTS Round
            pulled_arm_adv2 = gpts_learner2.pull_arm()
            reward_adv2 = env_bid2.round(pulled_arm_adv2, conv_rate=convertion_rate2[pulled_arm_price2], price=prices2[pulled_arm_price2])
            gpts_learner2.update(pulled_arm_adv2, reward_adv2)
            
            # GPUCB Round
            pulled_arm_adv = gpucb_learner2.pull_arm()
            reward_adv2 = env_bid2.round(pulled_arm_adv2, conv_rate=convertion_rate2[pulled_arm_price2], price=prices2[pulled_arm_price2])
            gpucb_learner2.update(pulled_arm_adv2, reward_adv2)
            
            
            # Store collected rewards for each learner
            gpts_rewards_per_experiment2.append(gpts_learner2.collected_rewards)
            gpucb_rewards_per_experiment2.append(gpucb_learner2.collected_rewards)
                      
        
        
        elif select_class==2:
    
            # Price TS Round (We decide a price)
            pulled_arm_price3 = ts_learner_price2.pull_arm()
            reward3 = env_price3.next_round(pulled_arm_price3)
            reward_com3 = reward3*prices3[pulled_arm_price3]
            ts_learner_price2.update(pulled_arm_price3, reward_com3)
            
            # GPTS Round
            pulled_arm_adv2 = gpts_learner2.pull_arm()
            reward_adv2 = env_bid3.round(pulled_arm_adv2, conv_rate=convertion_rate3[pulled_arm_price3], price=prices3[pulled_arm_price3])
            gpts_learner2.update(pulled_arm_adv2, reward_adv2)
            
            # GPUCB Round
            pulled_arm_adv2 = gpucb_learner2.pull_arm()
            reward_adv2 = env_bid3.round(pulled_arm_adv2, conv_rate=convertion_rate3[pulled_arm_price3], price=prices3[pulled_arm_price3])
            gpucb_learner2.update(pulled_arm_adv2, reward_adv2)
            
            # Store collected rewards for each learner
            gpts_rewards_per_experiment2.append(gpts_learner2.collected_rewards)
            gpucb_rewards_per_experiment2.append(gpucb_learner2.collected_rewards)




# Pad the arrays with the mean
def pad_arrays_with_mean(arrays):
    max_length = max(len(arr) for arr in arrays)
    padded_arrays = [np.pad(arr, (0, max_length - len(arr)), mode='constant', constant_values=np.mean(arr)) for arr in arrays]
    return np.array(padded_arrays)

# Pad the arrays with the mean for GP-TS rewards
gpts_rewards_per_experiment1 = pad_arrays_with_mean(gpts_rewards_per_experiment1)
gpts_rewards_per_experiment2 = pad_arrays_with_mean(gpts_rewards_per_experiment2)


# Pad the arrays with the mean for GP-UCB rewards
gpucb_rewards_per_experiment1 = pad_arrays_with_mean(gpucb_rewards_per_experiment1)
gpucb_rewards_per_experiment2 = pad_arrays_with_mean(gpucb_rewards_per_experiment2)


print('Optimal Profit: ',opt_profit1*T)
print('GP-TS Profit: ',max(np.cumsum(np.mean(np.array(gpts_rewards_per_experiment1),axis=0))))
print('GP-UCB Profit: ',max(np.cumsum(np.mean(np.array(gpucb_rewards_per_experiment1),axis=0))))

print('Optimal Profit: ',opt_profit2*T)
print('GP-TS Profit: ',max(np.cumsum(np.mean(np.array(gpts_rewards_per_experiment2),axis=0))))
print('GP-UCB Profit: ',max(np.cumsum(np.mean(np.array(gpucb_rewards_per_experiment2),axis=0))))


# C1

# Plot the cumulative regret
plt.figure()
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt_profit1 - np.array(gpts_rewards_per_experiment1),axis=0)),'r')
plt.plot(np.cumsum(np.mean(opt_profit1 - np.array(gpucb_rewards_per_experiment1),axis=0)),'g')
plt.legend(["GP-TS","GP-UCB"])
plt.show()



# plot the cumulative profit
plt.figure('Profit')
plt.plot(np.cumsum(np.mean(np.array(gpts_rewards_per_experiment1),axis=0)),'r')
plt.plot(np.cumsum(np.mean(np.array(gpucb_rewards_per_experiment1),axis=0)),'g')
plt.plot(np.cumsum(np.full(T, opt_profit1)), 'b')
plt.xlabel("t")
plt.ylabel("Profit")
plt.legend(["GP-TS","GP-UCB","Optimal"])
plt.show()


# plot the Instantaneous Regret
plt.figure("Instantaneous Regret")
plt.xlabel("t")
plt.ylabel("Instantaneous Regret")
plt.plot(np.mean(opt_profit1 - np.array(gpts_rewards_per_experiment1),axis=0),'r')
plt.plot(np.mean(opt_profit1 - np.array(gpucb_rewards_per_experiment1),axis=0),'g')
plt.legend(["GP-TS","GP-UCB"])

# plot the Instantaneous profit
plt.figure('Instantaneous Reward')
plt.plot(np.mean(np.array(gpts_rewards_per_experiment1),axis=0),'r')
plt.plot(np.mean(np.array(gpucb_rewards_per_experiment1),axis=0),'g')
plt.plot(np.full(T, opt_profit1), 'b')
plt.xlabel("t")
plt.ylabel("Profit")
plt.legend(["GP-TS","GP-UCB","Optimal"])
plt.show()


# C2

# Plot the cumulative regret
plt.figure()
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt_profit2 - np.array(gpts_rewards_per_experiment2),axis=0)),'r')
plt.plot(np.cumsum(np.mean(opt_profit2 - np.array(gpucb_rewards_per_experiment2),axis=0)),'g')
plt.legend(["GP-TS","GP-UCB"])
plt.show()


# plot the cumulative profit
plt.figure('Profit')
plt.plot(np.cumsum(np.mean(np.array(gpts_rewards_per_experiment2),axis=0)),'r')
plt.plot(np.cumsum(np.mean(np.array(gpucb_rewards_per_experiment2),axis=0)),'g')
plt.plot(np.cumsum(np.full(T, opt_profit2)), 'b')
plt.xlabel("t")
plt.ylabel("Profit")
plt.legend(["GP-TS","GP-UCB","Optimal"])
plt.show()

# plot the Instantaneous Regret
plt.figure("Instantaneous Regret")
plt.xlabel("t")
plt.ylabel("Instantaneous Regret")
plt.plot(np.mean(opt_profit2 - np.array(gpts_rewards_per_experiment2),axis=0),'r')
plt.plot(np.mean(opt_profit2 - np.array(gpucb_rewards_per_experiment2),axis=0),'g')
plt.legend(["GP-TS","GP-UCB"])

# plot the Instantaneous profit
plt.figure('Instantaneous Reward')
plt.plot(np.mean(np.array(gpts_rewards_per_experiment2),axis=0),'r')
plt.plot(np.mean(np.array(gpucb_rewards_per_experiment2),axis=0),'g')
plt.plot(np.full(T, opt_profit2), 'b')
plt.xlabel("t")
plt.ylabel("Profit")
plt.legend(["GP-TS","GP-UCB","Optimal"])
plt.show()




# C3

# Plot the cumulative regret
plt.figure()
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt_profit3 - np.array(gpts_rewards_per_experiment2),axis=0)),'r')
plt.plot(np.cumsum(np.mean(opt_profit3 - np.array(gpucb_rewards_per_experiment2),axis=0)),'g')
plt.legend(["GP-TS","GP-UCB"])
plt.show()


# plot the cumulative profit
plt.figure('Profit')
plt.plot(np.cumsum(np.mean(np.array(gpts_rewards_per_experiment2),axis=0)),'r')
plt.plot(np.cumsum(np.mean(np.array(gpucb_rewards_per_experiment2),axis=0)),'g')
plt.plot(np.cumsum(np.full(T, opt_profit3)), 'b')
plt.xlabel("t")
plt.ylabel("Profit")
plt.legend(["GP-TS","GP-UCB","Optimal"])
plt.show()

# plot the Instantaneous Regret
plt.figure("Instantaneous Regret")
plt.xlabel("t")
plt.ylabel("Instantaneous Regret")
plt.plot(np.mean(opt_profit - np.array(gpts_rewards_per_experiment2),axis=0),'r')
plt.plot(np.mean(opt_profit - np.array(gpucb_rewards_per_experiment2),axis=0),'g')
plt.legend(["GP-TS","GP-UCB"])

# plot the Instantaneous profit
plt.figure('Instantaneous Reward')
plt.plot(np.mean(np.array(gpts_rewards_per_experiment2),axis=0),'r')
plt.plot(np.mean(np.array(gpucb_rewards_per_experiment2),axis=0),'g')
plt.plot(np.full(T, opt_profit3), 'b')
plt.xlabel("t")
plt.ylabel("Profit")
plt.legend(["GP-TS","GP-UCB","Optimal"])
plt.show()