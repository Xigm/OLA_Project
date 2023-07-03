from BiddingEnvironment import Environment
from GPTS_ADV_Learner import GPTS_Learner
import matplotlib.pyplot as plt
from tqdm import tqdm 
import numpy as np

# i don't know why it gives me this error: ConvergenceWarning: The optimal value found for dimension 0 of parameter k2__length_scale is close to the specified lower bound 1e-05. Decreasing the bound and calling fit again may find a better value.
gains = lambda daily_clicks, conversion, margin, cumulative_daily_costs: daily_clicks * conversion * margin - cumulative_daily_costs

n_arms_price = 5
n_arms_adv = 10
p = np.array([0.5, 0.4, 0.3, 0.25, 0.2])
prices = np.array([10, 15, 20, 25, 30])
price_opt = np.max(p*prices)
p_long = np.array(p)
p_long = np.repeat(p_long, n_arms_adv) # i create an array with all the probabilities for all the combination with prices
min_bid = 0
max_bid = 1
bids = np.linspace(min_bid, max_bid, n_arms_adv)
sigma = 0
n_arms = n_arms_price*n_arms_adv
X, Y = np.meshgrid(bids, prices,)
arms = np.column_stack([X.ravel(), Y.ravel()]) # i create a matrix 50*2 with all the combinations prices, bids
print(arms.shape)
T = 365
n_experiments = 3

rewards_per_experiment = []


for e in tqdm(range(0,n_experiments)):
    env = Environment(n_arms=n_arms, arms=arms, sigma=sigma, prob=p_long)
    gpts_learner = GPTS_Learner(n_arms=n_arms, arms=arms)
    adv_daily_cost = []
    daily_rev = []

    for t in range(0,T):
        # Thompson Sampling Learner
        pulled_arm = gpts_learner.pull_arm()
        ts_rewards = env.round(pulled_arm)[0]
        gpts_learner.update(pulled_arm=pulled_arm, reward=ts_rewards)

    rewards_per_experiment.append(gpts_learner.collected_rewards) #add to the experiments reward the result obtained


price_opt = np.max(p*prices) # find the best price
opt = price_opt*env.best[1]-env.best[0]*env.best[1] # find the optimal profit for the combination price/bids
plt.figure(0)
plt.ylabel('Regret')
plt.xlabel('t')
plt.plot(np.cumsum(np.mean(opt[0] - rewards_per_experiment, axis=0)), 'r')
plt.legend(['GPTS'])
plt.show()