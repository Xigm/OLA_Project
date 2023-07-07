import numpy as np

class Learner():
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.t = 0
        self.rewards_per_arm = x = [[] for i in range(n_arms)] #create 4 array one for each arm where i can memorize the rewards
        self.collected_rewards = np.array([]) # create an array where i can memorize the rewards (of each experiment) but not distinguished by arm

    def update_observations(self, pulled_arm, reward):
        self.rewards_per_arm[pulled_arm].append(reward) #add the reward to the array of the pulled arm
        self.collected_rewards = np.append(self.collected_rewards, reward) #add reward to collected_rewards
