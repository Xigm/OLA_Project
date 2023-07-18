from step5.pricing import PricingModel
from step5.advertising import AdvertisingModel
from step5.ucb1 import UCB1
from step5.passive_ucb1 import PassiveUCB1
from step5.active_ucb1 import ActiveUCB1
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_results(avg_cumulative_regret_ucb1, avg_cumulative_regret_passive, avg_cumulative_regret_active,
                 avg_cumulative_reward_ucb1, avg_cumulative_reward_passive, avg_cumulative_reward_active,
                 avg_instantaneous_regret_ucb1, avg_instantaneous_regret_passive, avg_instantaneous_regret_active,
                 avg_instantaneous_reward_ucb1, avg_instantaneous_reward_passive, avg_instantaneous_reward_active,
                 save_dir=None):    #  specify the directory to save the plots
    phases = np.arange(1, len(avg_cumulative_regret_ucb1) + 1)

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.plot(phases, avg_cumulative_regret_ucb1, label='UCB1')
    plt.plot(phases, avg_cumulative_regret_passive, label='Passive UCB1')
    plt.plot(phases, avg_cumulative_regret_active, label='Active UCB1')
    plt.xlabel('Phase')
    plt.ylabel('Cumulative Regret')
    plt.legend()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'cumulative_regret.png'))
    else:
        plt.show()

    plt.subplot(2, 2, 2)
    plt.plot(phases, avg_cumulative_reward_ucb1, label='UCB1')
    plt.plot(phases, avg_cumulative_reward_passive, label='Passive UCB1')
    plt.plot(phases, avg_cumulative_reward_active, label='Active UCB1')
    plt.xlabel('Phase')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'cumulative_reward.png'))
    else:
        plt.show()

    plt.subplot(2, 2, 3)
    for phase in range(len(avg_instantaneous_regret_ucb1)):
        plt.plot(avg_instantaneous_regret_ucb1[phase], label=f'Phase {phase + 1}')
    plt.xlabel('Day')
    plt.ylabel('Instantaneous Regret')
    plt.legend()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'cumulative_regret_ins.png'))
    else:
        plt.show()  

    plt.subplot(2, 2, 4)
    for phase in range(len(avg_instantaneous_reward_ucb1)):
        plt.plot(avg_instantaneous_reward_ucb1[phase], label=f'Phase {phase + 1}')
    plt.xlabel('Day')
    plt.ylabel('Instantaneous Reward')
    plt.legend()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'cumulative_reward_ins.png'))
    else:
        plt.show()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    num_phases = 3
    # change for sensitvity analysis
    num_runs = 100
    window_size = 20
    change_threshold = 5

    pricing_model = PricingModel(num_phases)    # the curve related to the pricing problem is unknown
    pricing_model.generate_prices()

    advertising_model = AdvertisingModel(1)     # a single-user class C1
    advertising_model.generate_advertising_curves()     # curves related to the advertising problems are known

    ucb1 = UCB1(num_phases, num_runs)   # situation in which the curves related to pricing are non-stationary, being subject to seasonal phases (3 different phases spread over the time horizon)
    ucb1.run_algorithm(pricing_model, advertising_model)

    passive_ucb1 = PassiveUCB1(num_phases, num_runs, window_size)
    passive_ucb1.run_algorithm(pricing_model, advertising_model)

    active_ucb1 = ActiveUCB1(num_phases, num_runs, change_threshold)
    active_ucb1.run_algorithm(pricing_model, advertising_model)

    # report the plots with the average value and standard deviation of the cumulative regret, cumulative reward, instantaneous regret, and instantaneous reward
    plot_results(ucb1.avg_cumulative_regret, passive_ucb1.avg_cumulative_regret, active_ucb1.avg_cumulative_regret,
                 ucb1.avg_cumulative_reward, passive_ucb1.avg_cumulative_reward, active_ucb1.avg_cumulative_reward,
                 ucb1.avg_instantaneous_regret, passive_ucb1.avg_instantaneous_regret, active_ucb1.avg_instantaneous_regret,
                 ucb1.avg_instantaneous_reward, passive_ucb1.avg_instantaneous_reward, active_ucb1.avg_instantaneous_reward)
