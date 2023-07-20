# Non-Stationary Environmrnt
from environment import Environment
import numpy as np

class NSEnvironment(Environment):
    # it depends on time
    def __init__(self, n_arms, probabilities, horizon, n_phases, prices):
        super().__init__(n_arms, probabilities)
        self.t = 0
        self.phases_size = horizon/n_phases
        self.n_curves = np.array(probabilities).shape[0]
        self.prices = prices

    def round(self, pulled_arm):
        # returns the reward of the chosen arm
        current_phase = int((np.floor(self.t / self.phases_size))%self.n_curves)
        p = self.probabilities[current_phase][pulled_arm]
        reward = np.random.binomial(1, p)
        # reward should be affected by the price, as we try to maximize
        # gain: price*CR
        reward *= self.prices[pulled_arm]
        self.t += 1
        return reward

