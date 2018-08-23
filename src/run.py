from user_simulator.my_rule_based_user_sim import RuleBasedUserSimulator
from agent.rule_based_agent import RuleBasedAgent
from dialog_arbiter.arbiter import DialogArbiter
import dialog_config
import model_params
import numpy as np
import math
import statsmodels.api as sm
import pickle
import matplotlib.pyplot as plt


def get_cvm_divergence(true, model):
    ecdf_true = sm.distributions.ECDF(true)
    ecdf_model = sm.distributions.ECDF(model)
    y1 = ecdf_true(true)
    y2 = ecdf_model(true)
    N = np.shape(true)[0]
    alpha = math.sqrt(12*N/(4*N*N-1))
    D = alpha * math.sqrt(np.sum((y1-y2)**2))
    return D


def construct_edf(true, model):
    ecdf_true = sm.distributions.ECDF(true)
    ecdf_model = sm.distributions.ECDF(model)
    x = np.linspace(0, 7)
    y_true = ecdf_true(x)
    y_model = ecdf_model(x)
    print(x)
    f = plt.figure()
    plt.step(x, y_true, label='True')
    plt.step(x, y_model, label='Synthetic')
    plt.legend(loc=4)
    plt.xlabel('Rapport')
    plt.ylabel('Empirical Distribution Function (EDF)')
    f.savefig('rapport_edf.pdf', bbox_inches='tight')
    plt.show()


def get_pmf(p):
    return p/np.sum(p)


def get_KL_div(p, q):
    epsilon = 1e-3
    p[p == 0] = epsilon
    q[q == 0] = epsilon
    return np.sum(p * np.log(p) - p * np.log(q))


def get_KL_distance(p, q):
    return 0.5 * (get_KL_div(p, q) + get_KL_div(q, p))


def run_dialog(num_dialogs, arbiter):
    cumulative_reward = 0

    turn_list = []
    successes = 0
    cs_dist = None

    total_recos = 0
    total_accepted = 0

    rapp_vals = {}
    for cluster_id in model_params.all_clusters:
        rapp_vals[cluster_id] = [[], []]

    kl_list = []
    cvm_list = []
    all_rapps_gold = pickle.load(open(dialog_config.data_path + 'all_rapps_gold_full.pkl', 'rb'))
    true_dist_pmf = get_pmf(np.array([354., 82., 86., 8., 25., 1101.])[np.newaxis, :])

    for it in range(dialog_config.max_iter):
        print("###########################################################")
        print("Iteration: ", it)
        print("###########################################################")
        for dialog in range(num_dialogs):
            arbiter.initialize()
            dialog_over = False

            while not dialog_over:
                reward, dialog_over, state = arbiter.next()

            total_recos += state['session'] + state['person']
            total_accepted += state['num_accepted']['session']['send_msg_tlink'] + state['num_accepted']['person']['send_msg_tlink']

            if cs_dist is None:
                cs_dist = state['cs_dist_user']
            else:
                cs_dist += state['cs_dist_user']

            if dialog_config.all_together:
                user_index = 'all'
            else:
                user_index = 0
                if arbiter.user.type['rapport_care']:
                    user_index = 1

            for i in [0, 1]:
                rapp_vals[user_index][i] += state['rapp'][i]

            if reward > 0:
                successes += 1
                cumulative_reward += reward

            turn = state['turn'] / 2 - 1
            turn_list.append(turn)

        all_rapps = []
        if dialog_config.all_together:
            for i in [0, 1]:
                all_rapps.extend(rapp_vals['all'][i])

        else:
            for i in [0]:
                for j in [0, 1]:
                    all_rapps.extend(rapp_vals[i][j])

        cs_dist_pmf = get_pmf(cs_dist)
        print("True distribution: ", true_dist_pmf)
        print("Predicted distribution: ", cs_dist_pmf)
        kl_dist = get_KL_distance(true_dist_pmf, cs_dist_pmf)
        cvm = get_cvm_divergence(np.array(all_rapps_gold), np.array(all_rapps))
        print("KL distance: %.3f" % kl_dist)
        print("CVM divergence: %.3f" % cvm)
        kl_list.append(kl_dist)
        cvm_list.append(cvm)

    print("###########################################################")
    print("KL distance: ", end='')
    print("Mean: %.3f, " % np.average(kl_list), end='')
    print("Std. Dev.: %.3f" % np.std(kl_list))

    print("CvM divergence: ", end='')
    print("Mean: %.3f, " % np.average(cvm_list), end='')
    print("Std. Dev.: %.3f" % np.std(cvm_list))
    print("###########################################################")


if __name__ == "__main__":
    # Slot sets and act sets
    slot_set = dialog_config.slot_set
    phase_set = dialog_config.phase_set
    agent_act_set = dialog_config.sys_act_set
    user_act_set = dialog_config.user_act_set

    agent = RuleBasedAgent()
    user = RuleBasedUserSimulator()

    param_state_tracker = {}
    pass
    param_state_tracker['count_slots'] = dialog_config.count_slots
    param_state_tracker['reward_slots'] = dialog_config.reward.keys()
    arbiter = DialogArbiter(user, agent, slot_set, param_state_tracker)

    run_dialog(dialog_config.num_dialogs, arbiter)