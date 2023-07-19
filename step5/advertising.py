import numpy as np

class AdvertisingModel:
    # an implementation of the AdvertisingModel class, which is used to generate and manipulate advertising curves for different user classes
    def __init__(self, num_classes):
        self.num_classes = num_classes  #  the number of user classes for which advertising curves will be generated
        self.advertising_curves = None
    
    def generate_advertising_curves(self):
        # generate advertising curves for each class
        # these random values represent the advertising curves, which describe the relationship between advertising investment (bid) and the resulting number of clicks or cumulative cost

        self.advertising_curves = np.random.uniform(1, 10, (self.num_classes, 365))

    def get_clicks(self, class_id, bid):
        # returns the number of clicks for the given user class and bid

        # get the number of clicks for a specific class and bid
        clicks = self.advertising_curves[class_id] * bid

        # add Gaussian noise to the clicks to introduce variability
        clicks += np.random.normal(0, 1, clicks.shape)

        return clicks
    
    def get_cumulative_cost(self, class_id, bid):
        # calculates the cumulative daily click cost for the specified user class and bid

        # get the cumulative daily click cost for a specific class and bid
        cumulative_cost = np.cumsum(self.advertising_curves[class_id] * bid)

        # add Gaussian noise to the cumulative cost
        cumulative_cost += np.random.normal(0, 1, cumulative_cost.shape)

        return cumulative_cost
