from Learner import *

class UCB1_Learner(Learner):
    # always selects the arm with the upper confidence bound
    # the selected arm is the one that has the maximum value of the average of the samples 

    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.expected_rewards = np.zeros(n_arms) # expected reward for each arm
        self.count_pulled_arms = np.zeros(n_arms) # count the num of pulls for each arm
        self.upper_conf_bound = np.zeros(n_arms) # select arm that has the higher upper confidence bound reward

    def pull_arm(self):
        # function that selects which arm to pull at each round (time)
        if self.t < self.n_arms:
            # UCB1 samples the first time all the arms, then the arm with the higher upper confidence bound
            return self.t

        # compute upper confidence bounds for all arms
        for i in range(self.n_arms):
            confidence = np.sqrt(2 * np.log(self.t) / (self.count_pulled_arms[i]))
            self.upper_conf_bound[i] = self.expected_rewards[i] + confidence

        # returns the index of the arm with the highest upper confidence bound
        idxs = np.argmax(self.upper_conf_bound)
        return idxs

    def update(self, pulled_arm, reward):
        self.t += 1 # updates the time
        self.count_pulled_arms[pulled_arm] += 1 # updates the number of times an arm has been pulled
        self.update_observations(pulled_arm, reward) # updates the reward per arm and the rewards of all collected arms

        # updates the array of expected rewards that is the average of collected expected rewards for this arm
        self.expected_rewards[pulled_arm] = (self.expected_rewards[pulled_arm] * (self.t - 1) + reward) / self.t