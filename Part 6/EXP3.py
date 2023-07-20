import numpy as np
from Learner import Learner

class Exp3Algorithm(Learner):
    def __init__(self, num_arms, exploration_parameter):
        super().__init__(num_arms)
        self.num_arms = num_arms
        self.exploration_parameter = exploration_parameter
        self.weights = np.ones(num_arms)
        self.expected_rewards = np.zeros(num_arms) # expected reward for each arm
        self.count_pulled_arms = np.zeros(num_arms) # count the num of pulls for each arm
        self.t = 0


    def pull_arm(self):
        probabilities = self.weights / np.sum(self.weights)
        action = np.random.choice(self.num_arms, p=probabilities)
        return action


    def update_weights(self, action, reward):
        estimated_reward = reward / self.weights[action]
        exponent = self.exploration_parameter * estimated_reward / self.num_arms
        weight_update = np.exp(exponent)
        self.weights[action] *= weight_update

    def update(self, pulled_arm, reward):
        self.t += 1 # updates the time
        self.count_pulled_arms[pulled_arm] += 1 # updates the number of times an arm has been pulled
        self.update_observations(pulled_arm, reward) # updates the reward per arm and the rewards of all collected arms
        # updates the array of expected rewards that is the average of collected expected rewards for this arm
        self.expected_rewards[pulled_arm] = (self.expected_rewards[pulled_arm] * (self.t - 1) + reward) / self.t
        # update the weights of the exp3 algorithm
        self.update_weights(pulled_arm, reward)
