from src.user_simulator.my_rule_based_user_sim import RuleBasedUserSimulator
from src.agent.rule_based_agent import RuleBasedAgent
from src.dialog_arbiter.arbiter import DialogArbiter
from src import dialog_config
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd


def main():
    num_dialogs = 10
    user = RuleBasedUserSimulator()
    agent = RuleBasedAgent()
    slot_set = dialog_config.slot_set
    arbiter = DialogArbiter(user, agent)
    run_dialog(arbiter, num_dialogs, slot_set)


def construct_edf(true, model):
    ecdf_true = sm.distributions.ECDF(true)
    ecdf_model = sm.distributions.ECDF(model)
    x = np.linspace(min(model), max(model))
    y_true = ecdf_true(x)
    y_model = ecdf_model(x)
    print(x)
    f = plt.figure()
    plt.step(x, y_true, label='Training set')
    plt.step(x, y_model, label='Stochastic attitude towards rapport')
    plt.legend(loc=4)
    plt.xlabel('Dialog score')
    plt.ylabel('Empirical Distribution Function (EDF)')
    f.savefig('foo1.pdf', bbox_inches='tight')
    plt.show()


def get_cvm_divergence(true, model):
    ecdf_true = sm.distributions.ECDF(true)
    ecdf_model = sm.distributions.ECDF(model)
    y1 = ecdf_true(true)
    y2 = ecdf_model(true)
    N = np.shape(true)[0]
    alpha = math.sqrt(12*N/(4*N*N-1))
    D = alpha * math.sqrt(np.sum((y1-y2)**2))
    return D


def run_dialog(arbiter, num_dialogs, slot_set):
    cumulative_reward = 0
    count_slots = dialog_config.count_slots
    reward_slots = dialog_config.reward.keys()

    reward_list = []
    turn_list = []

    for dialog in range(num_dialogs):
        # print("Dialog: " + str(dialog))
        arbiter.initialize(slot_set)
        dialog_over = False
        while not dialog_over:
            reward, dialog_over, state = arbiter.next()

        # print()
        print("STATS:")
        print("Reward: " + str(reward))
        reward_list.append(reward)

        # print()
        turn = state['turn'] / 2 - 1
        print("Turns: " + str(turn))
        turn_list.append(turn)
        # print()

        # for c_slot in count_slots:
        #     print("Total " + c_slot + " recommendations: ", end="")
        #     print(state[c_slot])
        #     for r_slot in reward_slots:
        #         if r_slot == 'feedback':
        #             print("Feedback good count: " + str(state['num_accepted'][
        #                                                     c_slot][r_slot]))
        #         else:
        #             print("Top-link message count: " + str(
        #                 state['num_accepted'][c_slot][r_slot]))
        #     print()

    print(reward_list)
    print(turn_list)
    DF = pd.read_csv(dialog_config.true_reward_path)
    true_reward_list = DF.iloc[:, 0].values.tolist()
    print(true_reward_list)
    true = np.array(true_reward_list)
    model = np.array(reward_list)
    print(get_cvm_divergence(true, model))

    construct_edf(true, model)

    # DF = pd.read_csv(dialog_config.true_turn_path)
    # true_turn_list = DF.iloc[:, 0].values.tolist()
    # print(true_turn_list)
    # true = np.array(true_turn_list)
    # model = np.array(turn_list)
    # print(get_cvm_divergence(true, model))


if __name__ == "__main__":
    main()
