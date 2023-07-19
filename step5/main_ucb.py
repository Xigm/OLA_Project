import numpy as np
import matplotlib.pyplot as plt
from pricing import PricingModel
from advertising import AdvertisingModel
from passive_ucb1 import PassiveUCB1
from active_ucb1 import ActiveUCB1
from ucb1 import UCB1
import os

if __name__ == '__main__':
    print("Start...")
    num_phases = 3
    # change for sensitvity analysis
    num_runs = 100
    window_size = 20
    change_threshold = 30

    pricing_model = PricingModel(num_phases)    # the curve related to the pricing problem is unknown
    pricing_model.generate_prices()


    advertising_model = AdvertisingModel(1)     # a single-user class C1
    advertising_model.generate_advertising_curves()     # curves related to the advertising problems are known

# 1 start UCB1
    ucb1 = UCB1(num_phases, num_runs)   # situation in which the curves related to pricing are non-stationary, being subject to seasonal phases (3 different phases spread over the time horizon)
    ucb1.run_algorithm(pricing_model, advertising_model)

# 2 start Passive UCB1
    passive_ucb1 = PassiveUCB1(num_phases, num_runs, window_size)
    passive_ucb1.run_algorithm(pricing_model, advertising_model)

# 3 start Active UCB1
    active_ucb1 = ActiveUCB1(num_phases, num_runs, change_threshold)
    active_ucb1.run_algorithm(pricing_model, advertising_model)

# plots
##############################
    plt.figure(figsize=(16,9))
    plt.ylabel("Regret")
    plt.xlabel("t")
    plt.plot(np.cumsum(ucb1.cumulative_regret), color='r', label='UCB1')
    plt.plot(np.cumsum(passive_ucb1.cumulative_regret), color='g', label='Passive')
    plt.plot(np.cumsum(active_ucb1.cumulative_regret), color='b', label='Active')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join('.', 'cumulative_regret.png'))
    plt.show()

    # 2
    cumu_regret = np.cumsum(ucb1.cumulative_regret)
    cumu_passive_regret = np.cumsum(passive_ucb1.cumulative_regret)
    cumu_active_regret = np.cumsum(active_ucb1.cumulative_regret)

    # take the average over different runs
    avg_cumu_regret = np.mean(cumu_regret, axis=0)
    avg_cumu_passive_regret = np.mean(cumu_passive_regret, axis=0)
    avg_cumu_active_regret = np.mean(cumu_active_regret, axis=0)

    std_cumu_regret = np.std(cumu_regret, axis=0)
    std_cumu_passive_regret = np.std(cumu_passive_regret, axis=0)
    std_cumu_active_regret = np.std(cumu_active_regret, axis=0)

    plt.figure(figsize=(16,9))
    plt.ylabel("Regret")
    plt.xlabel("t")

    plt.plot(np.cumsum(ucb1.cumulative_regret), color='r', label='Regret')
    plt.plot(np.cumsum(passive_ucb1.cumulative_regret), color='g', label='Passive Regret')
    plt.plot(np.cumsum(active_ucb1.cumulative_regret), color='b', label='Active Regret')

    plt.plot(cumu_regret + 1 * std_cumu_regret / np.sqrt(num_runs), linestyle='--', color='r')
    plt.plot(cumu_regret - 1 * std_cumu_regret / np.sqrt(num_runs), linestyle='--', color='r')

    plt.plot(cumu_passive_regret + 1 * std_cumu_passive_regret / np.sqrt(num_runs), linestyle='--', color='g')
    plt.plot(cumu_passive_regret - 1 * std_cumu_passive_regret / np.sqrt(num_runs), linestyle='--', color='g')

    plt.plot(cumu_active_regret + 1 * std_cumu_active_regret / np.sqrt(num_runs), linestyle='--', color='b')
    plt.plot(cumu_active_regret - 1 * std_cumu_active_regret / np.sqrt(num_runs), linestyle='--', color='b')

    plt.legend()
    plt.grid()
    plt.savefig(os.path.join('.', 'cumulative_regret2.png'))
    plt.show()

##############################
    plt.figure(figsize=(16,9))
    plt.ylabel("Reward")
    plt.xlabel("t")
    plt.plot(np.cumsum(ucb1.cumulative_reward), color='r', label='UCB1')
    plt.plot(np.cumsum(passive_ucb1.cumulative_reward), color='g', label='Passive')
    plt.plot(np.cumsum(active_ucb1.cumulative_reward), color='b', label='Active')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join('.', 'cumulative_reward.png'))
    plt.show()

    # 2
    cumu_reward = np.cumsum(ucb1.cumulative_reward)
    cumu_passive_reward = np.cumsum(passive_ucb1.cumulative_reward)
    cumu_active_reward = np.cumsum(active_ucb1.cumulative_reward)

    avg_cumu_reward = np.mean(cumu_reward, axis=0)
    avg_cumu_passive_reward = np.mean(cumu_passive_reward, axis=0)
    avg_cumu_active_reward = np.mean(cumu_active_reward, axis=0)

    std_cumu_reward = np.std(cumu_reward, axis=0)
    std_cumu_passive_reward = np.std(cumu_passive_reward, axis=0)
    std_cumu_active_reward = np.std(cumu_active_reward, axis=0)

    plt.figure(figsize=(16,9))
    plt.ylabel("Reward")
    plt.xlabel("t")

    plt.plot(cumu_reward, color='r', label='Reward')
    plt.plot(cumu_passive_reward, color='g', label='Passive Reward')
    plt.plot(cumu_active_reward, color='b', label='Active Reward')

    plt.plot(cumu_reward + 1 * std_cumu_reward / np.sqrt(num_runs), linestyle='--', color='r')
    plt.plot(cumu_reward - 1 * std_cumu_reward / np.sqrt(num_runs), linestyle='--', color='r')

    plt.plot(cumu_passive_reward + 1 * std_cumu_passive_reward / np.sqrt(num_runs), linestyle='--', color='g')
    plt.plot(cumu_passive_reward - 1 * std_cumu_passive_reward / np.sqrt(num_runs), linestyle='--', color='g')

    plt.plot(cumu_active_reward + 1 * std_cumu_active_reward / np.sqrt(num_runs), linestyle='--', color='b')
    plt.plot(cumu_active_reward - 1 * std_cumu_active_reward / np.sqrt(num_runs), linestyle='--', color='b')

    plt.legend()
    plt.grid()
    plt.savefig(os.path.join('.', 'cumulative_reward2.png'))
    plt.show()

##############################
    plt.figure(figsize=(16,9))
    plt.ylabel("Instantaneous Regret")
    plt.xlabel("t")
    plt.plot(np.cumsum(ucb1.instantaneous_regret), color='r', label='UCB1')
    plt.plot(np.cumsum(passive_ucb1.instantaneous_regret), color='g', label='Passive')
    plt.plot(np.cumsum(active_ucb1.instantaneous_regret), color='b', label='Active')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join('.', 'instantaneous_regret.png'))
    plt.show()

    # 2
    cumu_regret = np.cumsum(ucb1.instantaneous_regret)
    cumu_passive_regret = np.cumsum(passive_ucb1.instantaneous_regret)
    cumu_active_regret = np.cumsum(active_ucb1.instantaneous_regret)

    avg_cumu_regret = np.mean(cumu_regret, axis=0)
    avg_cumu_passive_regret = np.mean(cumu_passive_regret, axis=0)
    avg_cumu_active_regret = np.mean(cumu_active_regret, axis=0)

    std_cumu_regret = np.std(cumu_regret, axis=0)
    std_cumu_passive_regret = np.std(cumu_passive_regret, axis=0)
    std_cumu_active_regret = np.std(cumu_active_regret, axis=0)

    plt.figure(figsize=(16,9))
    plt.ylabel("Instantaneous Regret")
    plt.xlabel("t")

    plt.plot(cumu_regret, color='r', label='Regret')
    plt.plot(cumu_passive_regret, color='g', label='Passive Regret')
    plt.plot(cumu_active_regret, color='b', label='Active Regret')

    plt.plot(cumu_regret + 1 * std_cumu_regret / np.sqrt(num_runs), linestyle='--', color='r')
    plt.plot(cumu_regret - 1 * std_cumu_regret / np.sqrt(num_runs), linestyle='--', color='r')

    plt.plot(cumu_passive_regret + 1 * std_cumu_passive_regret / np.sqrt(num_runs), linestyle='--', color='g')
    plt.plot(cumu_passive_regret - 1 * std_cumu_passive_regret / np.sqrt(num_runs), linestyle='--', color='g')

    plt.plot(cumu_active_regret + 1 * std_cumu_active_regret / np.sqrt(num_runs), linestyle='--', color='b')
    plt.plot(cumu_active_regret - 1 * std_cumu_active_regret / np.sqrt(num_runs), linestyle='--', color='b')

    plt.legend()
    plt.grid()
    plt.savefig(os.path.join('.', 'instantaneous_regret2.png'))
    plt.show()

##############################
    plt.figure(figsize=(16,9))
    plt.ylabel("Instantaneous Reward")
    plt.xlabel("t")
    plt.plot(np.cumsum(ucb1.instantaneous_reward), color='r', label='UCB1')
    plt.plot(np.cumsum(passive_ucb1.instantaneous_reward), color='g', label='Passive')
    plt.plot(np.cumsum(active_ucb1.instantaneous_reward), color='b', label='Active')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join('.', 'instantaneous_reward.png'))
    plt.show()

    # 2
    cumu_reward = np.cumsum(ucb1.instantaneous_reward)
    cumu_passive_reward = np.cumsum(passive_ucb1.instantaneous_reward)
    cumu_active_reward = np.cumsum(active_ucb1.instantaneous_reward)

    avg_cumu_reward = np.mean(cumu_reward, axis=0)
    avg_cumu_passive_reward = np.mean(cumu_passive_reward, axis=0)
    avg_cumu_active_reward = np.mean(cumu_active_reward, axis=0)

    std_cumu_reward = np.std(cumu_reward, axis=0)
    std_cumu_passive_reward = np.std(cumu_passive_reward, axis=0)
    std_cumu_active_reward = np.std(cumu_active_reward, axis=0)

    plt.figure(figsize=(16,9))
    plt.ylabel("Instantaneous Reward")
    plt.xlabel("t")

    plt.plot(cumu_reward, color='r', label='Reward')
    plt.plot(cumu_passive_reward, color='g', label='Passive Reward')
    plt.plot(cumu_active_reward, color='b', label='Active Reward')

    plt.plot(cumu_reward + 1 * std_cumu_reward / np.sqrt(num_runs), linestyle='--', color='r')
    plt.plot(cumu_reward - 1 * std_cumu_reward / np.sqrt(num_runs), linestyle='--', color='r')

    plt.plot(cumu_passive_reward + 1 * std_cumu_passive_reward / np.sqrt(num_runs), linestyle='--', color='g')
    plt.plot(cumu_passive_reward - 1 * std_cumu_passive_reward / np.sqrt(num_runs), linestyle='--', color='g')

    plt.plot(cumu_active_reward + 1 * std_cumu_active_reward / np.sqrt(num_runs), linestyle='--', color='b')
    plt.plot(cumu_active_reward - 1 * std_cumu_active_reward / np.sqrt(num_runs), linestyle='--', color='b')

    plt.legend()
    plt.grid()
    plt.savefig(os.path.join('.', 'instantaneous_reward2.png'))
    plt.show()

    print("Done!")




