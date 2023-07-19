# import numpy as np
from data import get_pricing_curve
import random
# class PricingModel:
#     def __init__(self, num_phases):
#         self.num_phases = num_phases
#         self.prices = None
#         self.probabilities = None
    
#     def generate_prices(self):
#         # use the pricing curve from data.py
#         list_of_dicts = get_pricing_curve()

#         # extract the prices and probabilities
#         prices_list = []
#         prob_list = []
#         for dict in list_of_dicts:
#             keys_list = list(dict.keys())
#             values_list = list(dict.values())
#             for key in keys_list:
#                 prices_list.append(key)
#             for value in values_list:
#                 prob_list.append(value)

#         prices_new = []
#         prob_new = []
#         for i in range(self.num_phases):
#             prices_new.append(prices_list[i])
#             prob_new.append(prob_list[i])

#         self.prices = prices_new    # prices_list
#         self.probabilities = prob_new   # prob_list
    
#     def get_price(self, phase):
#         # get the price for a single phase
#         return self.prices[phase % self.num_phases]

#     def change_pricing_curve(self, phase):
#         # simulate an abrupt change in the pricing curve for a specific phase
#         self.prices[phase % self.num_phases] += np.random.uniform(-10, 10)

#     def calculate_conversion_probability(self, price):
#         # probability based on the price 
#         for i in range(len(self.prices)):
#             if self.prices[i] == price:
#                 return self.probabilities[i]
#         return None



import numpy as np
import random

class PricingModel:
    def __init__(self, num_phases):
        self.num_phases = num_phases
        self.pricing_curves = None

    def generate_prices(self):
        # Generate pricing curves for each phase
        self.pricing_curves = []
        for _ in range(self.num_phases):
            curve = self._generate_price_curve()
            self.pricing_curves.append(curve)

    def get_price(self, phase, day):
        # Get the price for a specific phase and day
        curve = self.pricing_curves[phase]
        return curve[day]

    def _generate_price_curve(self):
        # use the pricing curve from data.py
        list_of_dicts = get_pricing_curve()

        # extract the prices and probabilities
        prices_list = []
        prob_list = []
        for dict in list_of_dicts:
            keys_list = list(dict.keys())
            values_list = list(dict.values())
            for key in keys_list:
                prices_list.append(key)
            for value in values_list:
                prob_list.append(value)

        prices_new = []
        prob_new = []
        for i in range(self.num_phases):
            prices_new.append(prices_list[i])
            prob_new.append(prob_list[i])

        self.prices = prices_new    # prices_list
        self.probabilities = prob_new   # prob_list

        # Generate prices for each day
        num_days = 365
        c1 = random.randint(-5, 5)
        c2 = random.randint(-5, 5)
        c3 = random.randint(-5, 5)
        
        prices = np.zeros(num_days)
        for day in range(num_days):
            # simulate an abrupt change in the pricing curve for a specific phase
            change_factor = 0
            if day <= 100:
                change_factor = c1
            if day <= 200 and day > 100:
                change_factor = c2
            if day <= 365 and day > 200:
                change_factor = c3

            price = random.choice(prices_new) + change_factor
            prices[day] = price

        return prices

