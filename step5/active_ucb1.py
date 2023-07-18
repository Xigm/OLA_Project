import numpy as np

class ActiveUCB1:
    def __init__(self, num_phases, num_runs, change_threshold):
        self.num_phases = num_phases
        self.num_runs = num_runs
        self.change_threshold = change_threshold    # exploits a change detection test
        self.avg_cumulative_regret = np.zeros(num_phases)
        self.avg_cumulative_reward = np.zeros(num_phases)
        self.avg_instantaneous_regret = np.zeros((num_phases, 365))
        self.avg_instantaneous_reward = np.zeros((num_phases, 365))

    def run_algorithm(self, pricing_model, advertising_model):
        for run in range(self.num_runs):
            detected_changes = [False] * self.num_phases

            for phase in range(self.num_phases):
                price = pricing_model.get_price(phase)
                conversion_prob = pricing_model.calculate_conversion_probability(price)
                clicks = advertising_model.get_clicks(0, price)
                cumulative_cost = advertising_model.get_cumulative_cost(0, price)

                optimal_reward = np.max(clicks * conversion_prob - cumulative_cost)
                optimal_price_index = np.argmax(clicks * conversion_prob - cumulative_cost)

                if phase > 0:
                    prev_optimal_reward = np.max(advertising_model.get_clicks(0, pricing_model.get_price(phase - 1)) * conversion_prob - advertising_model.get_cumulative_cost(0, pricing_model.get_price(phase - 1)))
                    if optimal_reward - prev_optimal_reward > self.change_threshold:
                        detected_changes[phase] = True

                if detected_changes[phase]:
                    pricing_model.change_pricing_curve(phase)

                self.avg_cumulative_regret[phase] += optimal_reward - (clicks[optimal_price_index] * conversion_prob - cumulative_cost[optimal_price_index])
                self.avg_cumulative_reward[phase] += clicks[optimal_price_index] * conversion_prob - cumulative_cost[optimal_price_index]

                self.avg_instantaneous_regret[phase] += optimal_reward - (clicks * conversion_prob - cumulative_cost)
                self.avg_instantaneous_reward[phase] += clicks * conversion_prob - cumulative_cost

        self.avg_cumulative_regret /= self.num_runs
        self.avg_cumulative_reward /= self.num_runs
        self.avg_instantaneous_regret /= self.num_runs
        self.avg_instantaneous_reward /= self.num_runs
