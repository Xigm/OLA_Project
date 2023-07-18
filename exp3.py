import numpy as np
import matplotlib.pyplot as plt
from ucb1 import UCB1
from active_ucb1 import ActiveUCB1
from passive_ucb1 import PassiveUCB1
from pricing import PricingModel
from advertising import AdvertisingModel
import os

def plot_results_exp3_ucb1(
    avg_cumulative_regret_exp3,
    avg_cumulative_reward_exp3,
    avg_cumulative_regret_ucb1,
    avg_cumulative_reward_ucb1,
    save_dir=None):

    phases = np.arange(1, len(avg_cumulative_regret_exp3) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(phases, avg_cumulative_regret_exp3, label='EXP3')
    plt.plot(phases, avg_cumulative_regret_ucb1, label='UCB1')
    plt.xlabel('Phases')
    plt.ylabel('Cumulative Regret')
    plt.title('EXP3 vs UCB1: Cumulative Regret Comparison')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(phases, avg_cumulative_reward_exp3, label='EXP3')
    plt.plot(phases, avg_cumulative_reward_ucb1, label='UCB1')
    plt.xlabel('Phases')
    plt.ylabel('Cumulative Reward')
    plt.title('EXP3 vs UCB1: Cumulative Reward Comparison')
    plt.legend()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'cumulative_regret.png'))
    else:
        plt.show()



class EXP3:
    def __init__(self, num_phases, num_runs, eta):
        self.num_phases = num_phases
        self.num_runs = num_runs
        self.eta = eta
        self.avg_cumulative_regret = np.zeros(num_phases)
        self.avg_cumulative_reward = np.zeros(num_phases)
        self.avg_instantaneous_regret = np.zeros((num_phases, num_runs))
        self.avg_instantaneous_reward = np.zeros((num_phases, num_runs))

    def select_arm(self, phase, weights):
        probabilities = np.exp(self.eta * weights[phase]) / np.sum(np.exp(self.eta * weights[phase]))
        arm = np.random.choice(len(weights[phase]), p=probabilities)
        return arm

    def update_weights(self, phase, arm, weights, reward):
        #print(weights)
        estimated_reward = reward / (self.eta * weights[phase][arm])
        #print(estimated_reward)
        weights[phase][arm] *= np.exp(self.eta * estimated_reward)

    def run(self, change_frequency):
        # pricing_model = PricingModel(self.num_phases)
        # pricing_model.generate_prices()
        # advertising_model = AdvertisingModel(1)
        # advertising_model.generate_advertising_curves() 

        num_arms = pricing_model.num_phases

        weights = np.ones((self.num_phases, num_arms))
        cumulative_reward = np.zeros((self.num_phases, num_arms))
        cumulative_regret = np.zeros((self.num_phases, num_arms))

        phase_duration = self.num_runs // self.num_phases
        phase_changes = np.arange(phase_duration, self.num_runs, phase_duration * change_frequency)
        phase_index = 0

        for t in range(self.num_runs):
            if t in phase_changes:
                phase_index = (phase_index + 1) % self.num_phases

            selected_arm = self.select_arm(phase_index, weights)
            price = pricing_model.get_price(phase_index)
            conversion_prob = pricing_model.calculate_conversion_probability(price)
            clicks = advertising_model.get_clicks(0, price)
            cumulative_cost = advertising_model.get_cumulative_cost(0, price)

            optimal_reward = np.max(clicks * conversion_prob - cumulative_cost)
            optimal_price_index = np.argmax(clicks * conversion_prob - cumulative_cost)

            self.update_weights(phase_index, selected_arm, weights, clicks[optimal_price_index] * conversion_prob - cumulative_cost[optimal_price_index])

            cumulative_reward[phase_index][selected_arm] += clicks[optimal_price_index] * conversion_prob - cumulative_cost[optimal_price_index]
            cumulative_regret[phase_index][selected_arm] += optimal_reward - (clicks[optimal_price_index] * conversion_prob - cumulative_cost[optimal_price_index])

        self.avg_cumulative_reward = np.mean(cumulative_reward, axis=1)
        self.avg_cumulative_regret = np.mean(cumulative_regret, axis=1)
        self.avg_instantaneous_reward = cumulative_reward / self.num_runs
        self.avg_instantaneous_regret = cumulative_regret / self.num_runs

        return self.avg_cumulative_regret, self.avg_cumulative_reward


if __name__ == "__main__":
    num_phases = 5
    num_runs = 100
    eta_exp3 = 0.1
    eta_ucb1 = 2.0
    change_frequency = 10

    pricing_model = PricingModel(num_phases)   
    pricing_model.generate_prices()

    advertising_model = AdvertisingModel(1)     # a single-user class C1
    advertising_model.generate_advertising_curves()     

    exp3 = EXP3(num_phases, num_runs, eta_exp3)
    ucb1 = UCB1(num_phases, num_runs)
    passive_ucb1 = PassiveUCB1(num_phases, num_runs, window_size=20)
    active_ucb1 = ActiveUCB1(num_phases, num_runs, change_threshold=5)

    # 1: simplified version of Step 5
    avg_cumulative_regret_exp3, avg_cumulative_reward_exp3 = exp3.run(change_frequency)
    avg_cumulative_regret_ucb1, avg_cumulative_reward_ucb1 = ucb1.run_algorithm(pricing_model, advertising_model)
    avg_cumulative_regret_passive_ucb1, avg_cumulative_reward_passive_ucb1 = passive_ucb1.run_algorithm(pricing_model, advertising_model)
    avg_cumulative_regret_active_ucb1, avg_cumulative_reward_active_ucb1 = active_ucb1.run_algorithm(pricing_model, advertising_model)

    plot_results_exp3_ucb1(
        avg_cumulative_regret_exp3,
        avg_cumulative_reward_exp3,
        avg_cumulative_regret_ucb1,
        avg_cumulative_reward_ucb1,
        save_dir='.'
    )

    # 2nd: higher non-stationarity degree
    num_phases = 5
    num_runs = 500
    eta_exp3 = 0.1
    eta_ucb1 = 2.0
    change_frequency = 2

    pricing_model.generate_prices()

    exp3 = EXP3(num_phases, num_runs, eta_exp3)
    ucb1 = UCB1(num_phases, num_runs)
    passive_ucb1 = PassiveUCB1(num_phases, num_runs, window_size=20)
    active_ucb1 = ActiveUCB1(num_phases, num_runs, change_threshold=5)

    avg_cumulative_regret_exp3, avg_cumulative_reward_exp3 = exp3.run(change_frequency)
    avg_cumulative_regret_ucb1, avg_cumulative_reward_ucb1 = ucb1.run_algorithm(pricing_model, advertising_model)
    avg_cumulative_regret_passive_ucb1, avg_cumulative_reward_passive_ucb1 = passive_ucb1.run_algorithm(pricing_model, advertising_model)
    avg_cumulative_regret_active_ucb1, avg_cumulative_reward_active_ucb1 = active_ucb1.run_algorithm(pricing_model, advertising_model)

    plot_results_exp3_ucb1(
        avg_cumulative_regret_exp3,
        avg_cumulative_reward_exp3,
        avg_cumulative_regret_ucb1,
        avg_cumulative_reward_ucb1,
        save_dir='.'
    )

# 59: RuntimeWarning: divide by zero encountered in scalar divide estimated_reward = reward / (self.eta * weights[phase][arm])
# Traceback (most recent call last):
#   File "/Users/anadrmic/Desktop/POLIMI/OLA/OLA_Project-main/part 5/exp3.py", line 125, in <module>
#     avg_cumulative_regret_ucb1, avg_cumulative_reward_ucb1 = ucb1.run_algorithm(pricing_model, advertising_model)