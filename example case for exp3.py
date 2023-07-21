import random

class EXP3:
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.weights = [1.0] * num_arms

    def select_arm(self):
        total_weight = sum(self.weights)
        probabilities = [weight / total_weight for weight in self.weights]
        arm = random.choices(range(self.num_arms), probabilities)[0]
        return arm

    def update_weights(self, arm, reward, gamma):
        estimated_reward = reward / max(gamma, probabilities[arm])
        self.weights[arm] *= math.exp((estimated_reward / self.num_arms))

# Example usage
num_arms = 5
exp3 = EXP3(num_arms)
num_iterations = 1000

for _ in range(num_iterations):
    arm = exp3.select_arm()
    # Simulate reward
    reward = random.uniform(0, 1)
    gamma = random.uniform(0, 1)
    exp3.update_weights(arm, reward, gamma)
