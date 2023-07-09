import numpy as np

# function that calculates the profit generated
def gain(x, marg, rate, clicks, cost):
    return marg * rate * clicks[x] - cost[x]

class BiddingEnvironment() : 
    def __init__(self, bids, sigma, clicks, cost):
        self.bids = bids
        self.clicks = clicks
        self.cost = cost
        self.sigmas = np.ones(len(bids)) * sigma

    def round(self, pulled_arm, conv_rate, price):
        return np.random.normal(gain(self.bids[pulled_arm], price, conv_rate, self.clicks, self.cost), self.sigmas[pulled_arm])
    
        