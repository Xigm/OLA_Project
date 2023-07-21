import numpy as np

class BiddingEnvironment() : 
    def __init__(self, bids, sigma, clicks, cost):
        self.bids = bids
        self.clicks = clicks
        self.cost = cost
        self.sigmas = np.ones(len(bids)) * sigma

    def round(self, pulled_arm, conv_rate, price):
        number_of_clicks = int(np.random.normal(self.clicks[self.bids[pulled_arm]], self.sigmas[pulled_arm]))
        reward = 0
        for _ in range(number_of_clicks):
            purchase = np.random.binomial(1, conv_rate)
            reward += purchase*price
        return reward - self.cost[self.bids[pulled_arm]]
   
    
        
