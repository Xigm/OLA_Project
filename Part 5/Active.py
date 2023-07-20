from UCB1_Learner import UCB1_Learner
import numpy as np
import math

class Active(UCB1_Learner):
    # periodically test for changes in the arms' reward distributions using a change detection test
    # when a change is detected, the algorithm updates its estimates of the arms' reward distributions and resets the exploration-exploitation strategy to explore the arms again to adapt to the new environment

    def __init__(self, n_arms, change_detection_window):
        super().__init__(n_arms)
        self.window_size = change_detection_window
        self.window_rewards_per_arm = [[] for _ in range(n_arms)]
        self.change_detected = False
        self.change_threshold = math.sqrt(math.log(n_arms) / change_detection_window)

    def pull_arm(self):
        if self.t < self.n_arms or self.change_detected:
            # if not enough data or a change is detected, explore all arms equally
            return self.t

        for i in range(self.n_arms):
            confidence = np.sqrt(2 * np.log(self.t) / (self.count_pulled_arms[i]))
            self.upper_conf_bound[i] = self.expected_rewards[i] + confidence

        idxs = np.argmax(self.upper_conf_bound)
        return idxs
    
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

        # check for change using CUSUM test

        if self.t > self.n_arms:
            for i in range(self.n_arms):
                cusum = 0
                if len(self.window_rewards_per_arm[i]) == self.window_size:
                    avg_reward = np.mean(self.window_rewards_per_arm[i])
                    cusum = np.maximum(0, cusum + (avg_reward - self.expected_rewards[i] - self.change_threshold))
                    if cusum > self.change_threshold:
                        self.change_detected = True
                        break
                else:
                    cusum = 0

        if self.change_detected:
            # reset exploration-exploitation strategy if a change is detected
            self.expected_rewards.fill(0)
            self.count_pulled_arms.fill(0)
            self.upper_conf_bound.fill(0)
            self.change_detected = False

