# import numpy as np
# from pricing import PricingModel
# from advertising import AdvertisingModel

# class PassiveUCB1:
#     def __init__(self, num_arms, num_runs, window_size):
#         self.num_arms = num_arms
#         self.num_runs = num_runs
#         self.window_size = window_size
#         self.num_pulls = np.zeros(num_arms)
#         self.mean_rewards = np.zeros(num_arms)
#         self.cumulative_regret = np.zeros(num_runs)
#         self.cumulative_reward = np.zeros(num_runs)
#         self.instantaneous_regret = np.zeros((num_runs, 365))
#         self.instantaneous_reward = np.zeros((num_runs, 365))

#     def select_arm(self):
#         exploration_term = np.sqrt(2 * np.log(np.sum(self.num_pulls) + 1) / self.num_pulls)
#         ucb_values = self.mean_rewards + exploration_term
#         selected_arm = np.argmax(ucb_values)
#         return selected_arm

#     def update_reward(self, arm, reward):
#         self.num_pulls[arm] += 1
#         self.cumulative_regret += np.max(self.mean_rewards) - reward
#         self.cumulative_reward += reward
#         self.mean_rewards[arm] = ((self.mean_rewards[arm] * (self.num_pulls[arm] - 1) + reward) / self.num_pulls[arm])

#     def run_algorithm(self, pricing_model, advertising_model):
#         for run in range(self.num_runs):
#             for arm in range(self.num_arms):
#                 price = pricing_model.get_price(arm)
#                 conversion_prob = pricing_model.calculate_conversion_probability(price)
#                 clicks = advertising_model.get_clicks(0, price)
#                 cumulative_cost = advertising_model.get_cumulative_cost(0, price)
#                 reward = np.max(clicks * conversion_prob - cumulative_cost)
#                 self.update_reward(arm, reward)

#             for t in range(self.num_arms, 365):
#                 arm = self.select_arm()
#                 price = pricing_model.get_price(arm)
#                 conversion_prob = pricing_model.calculate_conversion_probability(price)
#                 clicks = advertising_model.get_clicks(0, price)
#                 cumulative_cost = advertising_model.get_cumulative_cost(0, price)
#                 reward = np.max(clicks * conversion_prob - cumulative_cost)
#                 self.update_reward(arm, reward)

#                 if t >= self.window_size:
#                     regret = np.max(self.mean_rewards) - reward
#                     self.instantaneous_regret[run][t - self.window_size] = regret
#                     self.instantaneous_reward[run][t - self.window_size] = reward

#         self.cumulative_regret = np.cumsum(self.instantaneous_regret, axis=1)
#         self.cumulative_regret = np.mean(self.cumulative_regret, axis=0)
#         self.cumulative_reward /= self.num_runs

import numpy as np
from pricing import PricingModel

class PassiveUCB1:
    def __init__(self, num_phases, num_runs, window_size):
        self.num_phases = num_phases
        self.num_runs = num_runs
        self.window_size = window_size
        self.cumulative_regret = np.zeros((num_phases, num_runs))
        self.cumulative_reward = np.zeros((num_phases, num_runs))
        self.instantaneous_regret = np.zeros((num_phases, num_runs, 365))
        self.instantaneous_reward = np.zeros((num_phases, num_runs, 365))

    def run_algorithm(self, pricing_model, advertising_model):
        for phase in range(self.num_phases):
            for run in range(self.num_runs):
                pricing_model.generate_prices()  # Generate pricing curves for each phase
                self._run_phase(pricing_model, advertising_model, phase, run)

    def _run_phase(self, pricing_model, advertising_model, phase, run):
        num_arms = pricing_model.num_phases
        num_days = 365

        # Initialize variables for the phase
        num_clicks = np.ones(num_arms)
        total_reward = 0
        best_arm = 0
        prices_window = np.zeros(self.window_size)

        for day in range(num_days):
            # Choose an arm to pull
            arm = self._select_arm(pricing_model, phase, day, num_clicks, prices_window, run)

            # Get the price and advertising clicks for the chosen arm
            price = pricing_model.get_price(phase, day)
            clicks = advertising_model.get_clicks(0, price)

            # Update cumulative regret and reward
            regret = pricing_model.get_price(best_arm, day) - price
            self.cumulative_regret[phase, run] += regret
            self.cumulative_reward[phase, run] += price

            # Update instantaneous regret and reward
            self.instantaneous_regret[phase, run, day] = regret
            self.instantaneous_reward[phase, run, day] = price

            # Update total reward and number of clicks
            total_reward += price
            num_clicks[int(arm)] += np.sum(clicks)

            # Update the best arm based on the cumulative reward
            best_arm = np.argmax(num_clicks)

            # Update the prices window
            prices_window[:-1] = prices_window[1:]
            prices_window[-1] = price

    def _select_arm(self, pricing_model, phase, day, num_clicks, prices_window, run):
        num_arms = pricing_model.num_phases

        if np.sum(num_clicks) < num_arms:
            # Exploration phase: Pull each arm at least once
            arm = np.sum(num_clicks)
        else:
            # Exploitation phase: Use UCB1 algorithm with a sliding window
            avg_reward = self.cumulative_reward[phase, run] / num_clicks
            exploration_term = np.sqrt(2 * np.log(day + 1) / num_clicks)
            ucb_scores = avg_reward + exploration_term

            # Apply the sliding window to select the arm
            for win in prices_window:
                if win >= np.min(prices_window):
                    window_scores = ucb_scores 
                    arm = np.argmax(window_scores)

        return arm

