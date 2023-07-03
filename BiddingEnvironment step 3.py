import numpy as np

def fun_adv(x):
    return 100*(1.0 - np.exp(-4*x+3*x**3))

class Environment():
    def __init__(self, n_arms,  arms, sigma, prob):
        self.arms = arms
        self.bids = arms[:, 0]
        self.bids_unique = np.unique(self.bids)
        self.prices_unique = np.unique(self.arms[1])
        self.means = fun_adv(self.bids)
        self.sigmas = np.ones(len(self.bids))*sigma
        self.prob = prob
        self.best = [self.bids[np.where(self.means==np.max(self.means))[0]],np.max(self.means)]


    def round(self, pulled_arm):
        n_clicks = np.random.normal(self.means[pulled_arm], self.sigmas[pulled_arm])
        rev = 0
        for i in range(0,int(n_clicks)): # when the user click and go on the website then we have to calculate in how many cases the user will make the purchase given the price
            conv = np.random.binomial(1, self.prob[pulled_arm]) #it takes the convertion probability related to the selected price
            rev += conv*self.arms[pulled_arm][1]
        return rev-n_clicks*self.arms[pulled_arm][0], n_clicks, self.bids[pulled_arm] # it returns the profit so the function to maximize
