from learner_super import learner
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class GPTS_Learner(learner):
    def __init__(self, n_arms, arms):
        super().__init__(n_arms)
        self.arms = arms
        self.means = np.zeros(self.n_arms)
        self.sigmas = np.ones(self.n_arms)*100
        self.pulled_arms = np.empty((0, 2))# 2 represent the number of features in our case (bids, prices)
        alpha = 3 # here i don't know what to set
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=[1,30])#Here i have a doubt on the lenth scale because i tried three version and this one seems to be the best one but i don't have any idea
        self.gp = GaussianProcessRegressor(kernel, alpha=alpha**2, normalize_y=True, n_restarts_optimizer=9) # here i left it as it was

    def update_observations(self, arm_idx, reward):
        self.rewards_per_arm[arm_idx].append(reward)  # add the reward to the array of the pulled arm
        self.collected_rewards = np.append(self.collected_rewards, reward)
        row = np.array([[self.arms[arm_idx][0], self.arms[arm_idx][1]]]) # create the new line for the pulled_arms
        self.pulled_arms = np.vstack((self.pulled_arms, row))

    def update_model(self):
        x = self.pulled_arms
        y = self.collected_rewards
        self.gp.fit(x,y) # update the model with the new data
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms), return_std=True)
        self.sigmas = np.maximum(self.sigmas, 1e-2) # because sigma cannot be less than zero

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()

    def pull_arm(self):
        sampled_values = np.random.normal(self.means, self.sigmas)
        idx = np.argmax(sampled_values)
        return idx