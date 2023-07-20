import numpy as np
from environment import Environment
from UCB1_Learner import UCB1_Learner
import matplotlib.pyplot as plt
from ns_environment import NSEnvironment
from Passive import Passive
from Active import Active

n_arms = 5 # defined in the task
n_phases = 3

# the default pricing curve from data.py:
# [{10: 0.7, 25: 0.5, 40: 0.3, 55: 0.2, 70: 0.1}, {10: 0.8, 25: 0.6, 40: 0.5, 55: 0.4, 70: 0.2}, {10: 0.2, 25: 0.4, 40: 0.7, 55: 0.8, 70: 0.6}]
# columns: arms
# rows: phases
p = np.array([[0.7, 0.5, 0.3, 0.2, 0.1],
             [0.8, 0.6, 0.5, 0.4, 0.2],
             [0.2, 0.4, 0.7, 0.8, 0.6]])

T = 365 # the time horizon of 365 days
phases_len = int(T/n_phases)
n_experiments = 100

# store the reward for each learner:
ucb1_rewards_per_experiment = []
passive_rewards_per_experiment = []
active_rewards_per_experiment = []

window_size = int(T**0.5)
change_detection_window = int(T**0.5)

print("Experimenting...")

# iterate over each experiment:
for e in range(0, n_experiments):
    ucb1_env = NSEnvironment(n_arms, p, T)
    ucb1_learner = UCB1_Learner(n_arms)

    passive_env = NSEnvironment(n_arms, p, T)
    passive_learner = Passive(n_arms, window_size)

    active_env = NSEnvironment(n_arms, p, T)
    active_learner = Active(n_arms, change_detection_window)

    # iterate over each round (time) in the horizon (365 days)
    for t in range(0, T):

        # simulate the interaction between the environment and the learner
        pulled_arm = ucb1_learner.pull_arm()
        reward = ucb1_env.round(pulled_arm)
        ucb1_learner.update(pulled_arm, reward)

        pulled_arm = passive_learner.pull_arm()
        reward = passive_env.round(pulled_arm)
        passive_learner.update(pulled_arm, reward)

        pulled_arm = active_learner.pull_arm()
        reward = active_env.round(pulled_arm)
        active_learner.update(pulled_arm, reward)

    # at end of each e store the values of the collected rewards 
    ucb1_rewards_per_experiment.append(ucb1_learner.collected_rewards)
    passive_rewards_per_experiment.append(passive_learner.collected_rewards)
    active_rewards_per_experiment.append(active_learner.collected_rewards)

ucb1_instantaneus_regret = np.zeros(T)
passive_instantaneus_regret = np.zeros(T)
active_instantaneus_regret = np.zeros(T)

opt_per_phases = p.max(axis=1)
optimum_per_round = np.zeros(T)

# iterate over the phases
for i in range(n_phases):
    t_index = range(i*phases_len, (i+1)*phases_len)
    optimum_per_round[t_index] = opt_per_phases[i]

    ucb1_instantaneus_regret[t_index] = opt_per_phases[i] - np.mean(ucb1_rewards_per_experiment, axis=0)[t_index]
    passive_instantaneus_regret[t_index] = opt_per_phases[i] - np.mean(passive_rewards_per_experiment, axis=0)[t_index]
    active_instantaneus_regret[t_index] = opt_per_phases[i] - np.mean(active_rewards_per_experiment, axis=0)[t_index]



# plot the results:

plt.figure(0)

plt.plot(np.mean(ucb1_rewards_per_experiment, axis=0), 'r')
plt.plot(np.mean(passive_rewards_per_experiment, axis=0), 'b')
plt.plot(np.mean(active_rewards_per_experiment, axis=0), 'g')

plt.plot(optimum_per_round, 'k--')

plt.legend(['UCB1', 'Passive', 'Active', 'Optimum'])
plt.xlabel("t")
plt.ylabel("Reward")
plt.show()


plt.figure(1)

plt.plot(np.cumsum(ucb1_instantaneus_regret), 'r')
plt.plot(np.cumsum(passive_instantaneus_regret), 'b')
plt.plot(np.cumsum(passive_instantaneus_regret), 'g')

plt.legend(['UCB1', 'Passive', 'Active'])
plt.xlabel("t")
plt.ylabel("Regret")
plt.show()

print("Done!")

