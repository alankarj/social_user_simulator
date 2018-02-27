from src.user_simulator.my_rule_based_user_sim import RuleBasedUserSimulator
from src.agent.rule_based_agent import RuleBasedAgent
from src.agent.agent_dqn import AgentDQN
from src.dialog_arbiter.arbiter import DialogArbiter
from src import dialog_config

import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import statsmodels.api as sm

sim_test = False


if __name__ == "__main__":
    print("Inside main function.")

num_dialogs = 1000
warm_start_epochs = 100
agt = 1  # 0: Rule-based, 1: Flat-RL, 2: Hierarchical RL

# Slot sets and act sets
slot_set = dialog_config.slot_set
print("Slot set: ", slot_set)
phase_set = dialog_config.phase_set
print("Phase set: ", phase_set)
agent_act_set = dialog_config.sys_act_set
print("Agent act set: ", agent_act_set)
user_act_set = dialog_config.user_act_set
print("User act set: ", user_act_set)
agent_cs_set = dialog_config.agent_cs_set
print("Agent CS set: ", agent_cs_set)
user_cs_set = dialog_config.user_cs_set
print("User CS set: ", user_cs_set)

# User Parameters:
param_user = {}
pass
param_user['user_goal_slots'] = dialog_config.user_goal_slots
param_user['user_type_slots'] = dialog_config.user_type_slots
param_user['prob_user_type'] = dialog_config.prob_user_type
param_user['small_penalty'] = dialog_config.small_penalty
param_user['large_penalty'] = dialog_config.large_penalty
param_user['decision_points'] = dialog_config.decision_points
param_user['prob_funcs'] = dialog_config.prob_funcs
param_user['threshold_cs'] = dialog_config.threshold_cs
param_user['count_slots'] = dialog_config.count_slots
param_user['reward_slots'] = dialog_config.reward.keys()
param_user['reward'] = dialog_config.reward
param_user['max_turns'] = dialog_config.max_turns-2
param_user['max_recos'] = dialog_config.max_recos
param_user['constraint_violation_penalty'] = dialog_config.constraint_violation_penalty
param_user['min_turns'] = dialog_config.min_turns

# Agent Parameters:
param_agent = {}
if agt == 1:
    param_agent['slot_set'] = slot_set
    param_agent['phase_set'] = phase_set
    param_agent['user_act_set'] = user_act_set
    param_agent['agent_act_set'] = agent_act_set
    param_agent['user_cs_set'] = user_cs_set
    param_agent['agent_cs_set'] = agent_cs_set
    param_agent['feasible_actions'] = dialog_config.feasible_actions
    param_agent['reward_slots'] = dialog_config.reward.keys()
    param_agent['ERP_size_SL'] = 1000
    param_agent['ERP_size_SL'] = 100000
    param_agent['hidden_dim'] = 60
    param_agent['gamma'] = 0.9
    param_agent['predict_mode'] = False
    param_agent['warm_start'] = 1
    param_agent['epsilon'] = 1.0
    param_agent['trained_model_path'] = None
    param_agent['gamma'] = 0.9
    param_agent['bool_slots'] = dialog_config.bool_slots
    param_agent['max_turns'] = dialog_config.max_turns
    param_agent['max_recos'] = dialog_config.max_recos
    agent = AgentDQN(param_agent)
else:
    agent = RuleBasedAgent()


user = RuleBasedUserSimulator(param_user)
param_state_tracker = {}
pass
param_state_tracker['count_slots'] = dialog_config.count_slots
param_state_tracker['reward_slots'] = dialog_config.reward.keys()

arbiter = DialogArbiter(user, agent, slot_set, param_state_tracker)

#   Run num_episodes Conversation Simulations
status = {'successes': 0, 'count': 0, 'cumulative_reward': 0}

simulation_epoch_size = 100
batch_size = 16
warm_start = param_agent['warm_start']
success_rate_threshold = 0.7

""" Best Model and Performance Records """
best_res = {}
pass
best_res['success_rate'] = 0
best_res['ave_reward'] = float('-inf')
best_res['ave_turns'] = float('inf')
best_res['epoch'] = 0

# best_model = {}
# pass
# best_model['model'] = copy.deepcopy(agent)

perf_records = {}
pass
perf_records['success_rate'] = {}
perf_records['avg_turns'] = {}
perf_records['avg_reward'] = {}


def simulation_epoch():
    """ Run N simulation dialogs """
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0

    res = {}
    for dialog in range(simulation_epoch_size):
        arbiter.initialize()
        dialog_over = False

        while not dialog_over:
            reward, dialog_over, state = arbiter.next()

        cumulative_reward += reward

        if reward > 0:
            successes += 1

        cumulative_turns += int(state['turn'] * 0.5 - 1)

    res['success_rate'] = float(successes) / simulation_epoch_size
    res['avg_reward'] = float(cumulative_reward) / simulation_epoch_size
    res['avg_turns'] = float(cumulative_turns) / simulation_epoch_size
    print("Simulation success rate: %s, avg reward: %s, avg turns: %s" % (
        res['success_rate'], res['avg_reward'], res['avg_turns']))

    return res


def warm_start_simulation():
    """ Warm_Start Simulation (by Rule Policy) """
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0
    erp_full = False

    res = {}
    warm_start_run_epochs = 0

    for dialog in range(warm_start_epochs):
        arbiter.initialize()
        dialog_over = False

        while not dialog_over:
            reward, dialog_over, state = arbiter.next()
            if len(agent.experience_replay_pool) >= \
                    agent.experience_replay_pool_size:
                erp_full = True
                break

        if erp_full:
            break

        cumulative_reward += reward

        if reward > 0:
            successes += 1
            print("Warm start simulation dialog %s: Success" % dialog)
            print("Reward: ", reward)

        else:
            print("Warm start simulation dialog %s: Fail" % dialog)
            print("Reward: ", reward)

        cumulative_turns += int(state['turn'] * 0.5 - 1)
        warm_start_run_epochs += 1

    agent.warm_start = 2
    res['success_rate'] = float(successes) / warm_start_run_epochs
    res['avg_reward'] = float(cumulative_reward) / warm_start_run_epochs
    res['avg_turns'] = float(cumulative_turns) / warm_start_run_epochs
    print("Warm start: %s epochs, success rate: %s, avg reward: %s, ave turns: "
          "%s" % (warm_start_run_epochs, res['success_rate'], res['avg_reward'],
                  res['avg_turns']))

    print("Current experience replay buffer size %s" % (len(
        agent.experience_replay_pool)))


def run_dialog(num_dialogs, status):
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0

    if agt == 1 and param_agent['trained_model_path'] is None and warm_start \
            == 1:
        print('Starting warm start...')
        warm_start_simulation()

    print('Warm start finished, starting RL training now...')

    reward_list = []
    successes_list = []
    turns_list = []
    loss_list = []

    for dialog in range(num_dialogs):
        print("Epoch: %s" % dialog)

        arbiter.initialize()
        dialog_over = False

        while not dialog_over:
            reward, dialog_over, state = arbiter.next(print_info=True)

        cumulative_reward += reward
        turn = int(state['turn'] * 0.5 - 1)
        cumulative_turns += turn

        if reward > 0:
            successes += 1

        # Simulation
        if agt == 1 and param_agent['trained_model_path'] is None:
            agent.predict_mode = True
            simulation_res = simulation_epoch()

            perf_records['success_rate'][dialog] = simulation_res[
                'success_rate']
            perf_records['avg_turns'][dialog] = simulation_res['avg_turns']
            perf_records['avg_reward'][dialog] = simulation_res['avg_reward']

            if simulation_res['success_rate'] >= best_res['success_rate']:
                if simulation_res['success_rate'] >= success_rate_threshold:
                    print("Flushing out old shit.")
                    agent.experience_replay_pool = []
                    simulation_epoch()

            if simulation_res['success_rate'] > best_res['success_rate']:
                best_res['success_rate'] = simulation_res['success_rate']
                best_res['avg_reward'] = simulation_res['avg_reward']
                best_res['avg_turns'] = simulation_res['avg_turns']
                best_res['epoch'] = dialog

            agent.update_target_network()
            loss_list.append(agent.train(batch_size))
            agent.predict_mode = False

            agent.epsilon = max(agent.epsilon_min, agent.epsilon *
                                agent.epsilon_decay)

        reward_list.append(reward)
        turns_list.append(turn)
        successes_list.append(successes)

        print("Progress: %s / %s, Success rate: %s / %s Avg reward: %.2f Avg "
              "turns: %.2f" % (dialog + 1, num_dialogs, successes, dialog +
                               1, float(cumulative_reward) / (dialog + 1),
                               float(cumulative_turns) / (dialog + 1)))

    print("Success rate: %s / %s Avg reward: %.2f Avg turns: %.2f" % (
        successes, num_dialogs, float(cumulative_reward) / num_dialogs,
        float(cumulative_turns) / num_dialogs))
    print(loss_list)

    plt.plot(reward_list)
    plt.show()

    cumulative_reward = 0
    count_slots = dialog_config.count_slots
    reward_slots = dialog_config.reward.keys()

    reward_list = []
    turn_list = []
    successes = 0

    for dialog in range(num_dialogs):
        arbiter.initialize()
        dialog_over = False

        while not dialog_over:
            reward, dialog_over, state = arbiter.next()

        if reward > 0:
            successes += 1
        cumulative_reward += reward

        print("Reward: " + str(reward))
        reward_list.append(reward)

        turn = state['turn'] / 2 - 1
        print("Turns: " + str(turn))
        turn_list.append(turn)

    print("Fraction of successes: %f" % (successes/num_dialogs))
    print("Average reward: %f" % (cumulative_reward / num_dialogs))
    print("Max turn: %d" % max(turn_list))

    if sim_test:
        print(reward_list)
        print(turn_list)
        DF = pd.read_csv(dialog_config.true_reward_path)
        true_reward_list = DF.iloc[:, 0].values.tolist()
        print(true_reward_list)
        true = np.array(true_reward_list)
        model = np.array(reward_list)
        print(get_cvm_divergence(true, model))

        construct_edf(true, model)

        DF = pd.read_csv(dialog_config.true_turn_path)
        true_turn_list = DF.iloc[:, 0].values.tolist()
        print(true_turn_list)
        true = np.array(true_turn_list)
        model = np.array(turn_list)
        print(get_cvm_divergence(true, model))


def construct_edf(true, model):
    ecdf_true = sm.distributions.ECDF(true)
    ecdf_model = sm.distributions.ECDF(model)
    x = np.linspace(-30, 100)
    #x = np.linspace(min(model), max(model))
    y_true = ecdf_true(x)
    y_model = ecdf_model(x)
    print(x)
    f = plt.figure()
    plt.step(x, y_true, label='Training set')
    plt.step(x, y_model, label='P-type only (PO)')
    plt.legend(loc=4)
    plt.xlabel('Dialog score')
    plt.ylabel('Empirical Distribution Function (EDF)')
    f.savefig('foo3.pdf', bbox_inches='tight')
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


run_dialog(num_dialogs, status)
