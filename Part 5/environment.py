import numpy as np

class Environment():
    def __init__(self, n_arms, probabilities):
        self.n_arms = n_arms
        self.probabilities = probabilities

    def round(self, pulled_arm):
        # returns the reward of the chosen arm 
        reward = np.random.binomial(1, self.probabilities[pulled_arm]) # generates random samples from a binomial distribution
        # 1: the number of trials (or attempts)
        # self.probabilities[pulled_arm]: the probability of success for each trial
        # returns: either 0 or 1
        return reward