import numpy as np

class PassiveUCB1:
    def __init__(self, num_phases, num_runs, window_size):
        self.num_phases = num_phases
        self.num_runs = num_runs
        self.window_size = window_size  # exploits a sliding window
        self.avg_cumulative_regret = np.zeros(num_phases)
        self.avg_cumulative_reward = np.zeros(num_phases)
        self.avg_instantaneous_regret = np.zeros((num_phases, 365))
        self.avg_instantaneous_reward = np.zeros((num_phases, 365))

    def run_algorithm(self, pricing_model, advertising_model):
        for run in range(self.num_runs):
            window_rewards = np.zeros(self.window_size)

            for phase in range(self.num_phases):
                price = pricing_model.get_price(phase)
                conversion_prob = pricing_model.calculate_conversion_probability(price)
                clicks = advertising_model.get_clicks(0, price)
                cumulative_cost = advertising_model.get_cumulative_cost(0, price)

                optimal_reward = np.max(clicks * conversion_prob - cumulative_cost)
                optimal_price_index = np.argmax(clicks * conversion_prob - cumulative_cost)

                if phase >= self.window_size:
                    window_rewards[phase % self.window_size] = clicks[optimal_price_index] * conversion_prob - cumulative_cost[optimal_price_index]
                    window_optimal_reward = np.max(window_rewards)
                    self.avg_cumulative_regret[phase] += window_optimal_reward - (clicks[optimal_price_index] * conversion_prob - cumulative_cost[optimal_price_index])
                    self.avg_cumulative_reward[phase] += clicks[optimal_price_index] * conversion_prob - cumulative_cost[optimal_price_index]
                else:
                    self.avg_cumulative_regret[phase] += optimal_reward - (clicks[optimal_price_index] * conversion_prob - cumulative_cost[optimal_price_index])
                    self.avg_cumulative_reward[phase] += clicks[optimal_price_index] * conversion_prob - cumulative_cost[optimal_price_index]

                self.avg_instantaneous_regret[phase] += optimal_reward - (clicks * conversion_prob - cumulative_cost)
                self.avg_instantaneous_reward[phase] += clicks * conversion_prob - cumulative_cost

        self.avg_cumulative_regret /= self.num_runs
        self.avg_cumulative_reward /= self.num_runs
        self.avg_instantaneous_regret /= self.num_runs
        self.avg_instantaneous_reward /= self.num_runs
