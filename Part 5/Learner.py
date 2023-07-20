import numpy as np

class Learner:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.t = 0
        self.rewards_per_arm = [[] for i in range(n_arms)]
        # len(external list): the number of arms
        # len(internal list): the number of times we pull a specific arm
        self.collected_rewards = np.array([])

    def update_observations(self, pulled_arm, reward): 
        # the reward is given by the environment
        # now we have to update the observations:
        self.rewards_per_arm[pulled_arm].append(reward) # update list of the arm pulled with the reward given by the environmrnt
        self.collected_rewards = np.append(self.collected_rewards, reward) # keep the new reward in np array

