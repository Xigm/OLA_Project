from UCB1_Learner import UCB1_Learner
import numpy as np

class Passive(UCB1_Learner):
    # the only difference is that the confidence bound is computed using the last tau samples
    def __init__(self, n_arms, window_size):
        super().__init__(n_arms)
        self.window_size = window_size
        self.window_rewards_per_arm = [[] for _ in range(n_arms)]
        # self.pulled_arms = np.array([])

    def update(self, pulled_arm, reward):
        self.t += 1
        self.count_pulled_arms[pulled_arm] += 1
        self.update_observations(pulled_arm, reward)

        # add the reward to the sliding window for the selected arm
        self.window_rewards_per_arm[pulled_arm].append(reward)

        # check if the window size is exceeded for any arm and update statistics if needed
        for i in range(self.n_arms):
            if len(self.window_rewards_per_arm[i]) > self.window_size:
                removed_reward = self.window_rewards_per_arm[i].pop(0)
                self.expected_rewards[i] = (self.expected_rewards[i] * self.window_size - removed_reward) / (
                        self.window_size - 1)

        # update the array of expected rewards based on the rewards within the sliding window for each arm
        for i in range(self.n_arms):
            window_sum = np.sum(self.window_rewards_per_arm[i])
            window_count = len(self.window_rewards_per_arm[i])
            self.expected_rewards[i] = window_sum / window_count if window_count > 0 else 0

