from Learner_stp2 import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class GPUCB_Learner(Learner):
    def __init__(self, n_arms, arms):
        super().__init__(n_arms)
        self.arms = arms
        self.means = np.zeros(self.n_arms)
        self.sigmas = np.ones(self.n_arms) * 10
        self.pulled_arms = []
        self.num_pulls = np.ones(n_arms)
        alpha = 0.9
        kernel = C(1, (1e-3, 1e3)) * RBF(1, (1e-3, 1e3))
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True, n_restarts_optimizer=9)

    def update_observations(self, arm_idx, reward):
        super().update_observations(arm_idx, reward)
        self.pulled_arms.append(self.arms[arm_idx])

    def update_model(self):
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards
        self.gp.fit(x, y)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.sigmas = np.maximum(self.sigmas, 1e-2)

    def update(self, pulled_arm, reward):
        self.t += 1
        self.num_pulls[pulled_arm] +=1
        self.update_observations(pulled_arm, reward)
        self.update_model()

    def pull_arm(self):
        if self.t < self.n_arms:
            return self.t
        sampled_values = ([x + y * np.sqrt(2 * np.log(self.t) / (self.num_pulls[i]*(self.t-1))) for i, (x, y) in enumerate(zip(self.means, self.sigmas))])
        return np.argmax(sampled_values)