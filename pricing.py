import numpy as np

class PricingModel:
    def __init__(self, num_phases):
        self.num_phases = num_phases
        self.prices = None
    
    def generate_prices(self):
        # generate prices for each phase
        self.prices = np.random.uniform(1, 100, self.num_phases)
    
    def get_price(self, phase):
        # get the price for a single phase
        return self.prices[phase]

    def change_pricing_curve(self, phase):
        # simulate an abrupt change in the pricing curve for a specific phase
        self.prices[phase] += np.random.uniform(-10, 10)

    def calculate_conversion_probability(self, price):
        # calculate the conversion probability based on the price ?
        return 0.5

